"""Panel widgets over a plot's :class:`~vaft.plot.navigation.ControlState`.

The mapping follows the notebook controls
(:func:`vaft.plot.renderers.interactive._ipywidgets_controls`): each
:class:`~vaft.plot.controls.ControlSpec` becomes one widget of its ``kind``,
a widget change is handed to ``state.set`` (which validates it and redraws),
and a change made from code moves the widgets without re-firing them.  A
``multi`` control is a list of check boxes -- one click toggles one channel --
rather than a list box that needs Shift or Ctrl.
"""

from __future__ import annotations

from typing import Any, Callable

from ._require import require_panel

#: What a ``choice`` whose default is ``None`` shows: ``None`` means "leave
#: the option out", which none of the choices says.  Once another choice is
#: picked the state cannot hold ``None`` again (ControlSpec.validate), so
#: picking "(default)" afterwards is ignored and the widget moves back.
UNSET_LABEL = "(default)"
#: What the channel preset shows while individual channels replace it.
INDIVIDUAL_LABEL = "(individual channels)"


class _Individual:
    """The preset box's placeholder while individual channels are chosen."""

    def __repr__(self) -> str:  # pragma: no cover - shown only when debugging
        return INDIVIDUAL_LABEL


INDIVIDUAL = _Individual()


def _choice_options(control: Any) -> dict[str, Any]:
    labels = control.labels or tuple(map(str, control.options))
    return dict(zip(labels, control.options))


def _widget_value(control: Any, value: Any) -> Any:
    if control.kind == "multi":
        return list(value or ())
    if control.kind == "toggle":
        return bool(value)
    if control.kind == "text":
        return "" if value is None else str(value)
    return value


def _options_of(widget: Any) -> Any:
    options = getattr(widget, "options", None)
    return options if isinstance(options, dict) else {}


def _shown(state: Any, name: str) -> Any:
    """The value the widget for ``name`` shows for the current state."""
    if name == "selection" and state.values.get("channels"):
        return INDIVIDUAL
    return _widget_value(state.spec(name), state[name])


def panel_controls(
    state: Any,
    *,
    on_error: Callable[[Exception], Any] | None = None,
    scope: Callable[[], Any] | None = None,
) -> list[Any]:
    """Panel widgets for every control of ``state``, kept in step with it.

    A value the plot refuses to draw leaves the previous values in place; the
    widgets are moved back and ``on_error`` receives the exception.  Without
    ``on_error`` it propagates.  ``scope`` returns a context manager entered
    around every change, so the redraw it triggers draws under it (the
    reader's figure options, #1421).  The returned objects are what to lay out: a
    ``multi`` control is its check boxes inside a titled, scrolling column.
    """
    pn = require_panel()
    widgets: dict[str, Any] = {}
    #: name -> (player, value at a player position, player position of a value)
    players: dict[str, tuple[Any, Callable[[int], Any], Callable[[Any], int | None]]] = {}
    laid_out: list[Any] = []
    busy = {"on": False}
    has_channels = "channels" in {control.name for control in state.controls}

    def follow(current: Any, *, skip_players: bool = False) -> None:
        busy["on"] = True
        try:
            for name, widget in widgets.items():
                value = _shown(current, name)
                if value is None and UNSET_LABEL not in _options_of(widget):
                    # Only a choice with "(default)" among its options can show
                    # "left out"; the others keep their last value.
                    continue
                if widget.value != value:
                    widget.value = value
            for name, (player, _, position_of) in players.items():
                if skip_players:
                    continue
                position = position_of(current[name])
                if position is not None and player.value != position:
                    player.value = position
        finally:
            busy["on"] = False

    def apply_value(name: str, value: Any, *, from_player: bool = False) -> None:
        import contextlib

        before = state.values
        try:
            with scope() if scope is not None else contextlib.nullcontext():
                state.set(name, value)
        except Exception as error:  # the builder's refusal
            # The Matplotlib redraw restores the values itself; the Plotly
            # rebuild does not, so the values are put back here for both.
            state.restore(before)
            # A player keeps its position on a refused frame: moving it back
            # would make its next tick land on the same frame forever.
            follow(state, skip_players=from_player)
            if on_error is None:
                raise
            on_error(error)
        else:
            # A preset clears the channels (ControlState.set): the check
            # boxes and the preset box follow even when nothing redrew.
            follow(state)

    def setter(name: str) -> Callable[[Any], None]:
        def apply(event: Any) -> None:
            if busy["on"]:
                return
            if event.new is None or event.new is INDIVIDUAL:
                # A placeholder, not a value: show the state's value again.
                follow(state)
                return
            apply_value(name, event.new)
        return apply

    for control in state.controls:
        value = _shown(state, control.name)
        shown: Any = None
        if control.kind == "choice":
            options = _choice_options(control)
            if value is None:
                options = {UNSET_LABEL: None, **options}
            if control.name == "selection" and has_channels:
                options = {INDIVIDUAL_LABEL: INDIVIDUAL, **options}
            widget = pn.widgets.Select(label=control.label, options=options, value=value)
            watched = "value"
        elif control.kind == "multi":
            widget = pn.widgets.CheckBoxGroup(
                label=control.label, options=_choice_options(control), value=value, inline=False,
            )
            watched = "value"
            # A long channel list scrolls inside a fixed box instead of
            # pushing every later control off the sidebar.
            many = len(control.options) > 8
            shown = pn.Column(
                pn.pane.Markdown(control.label, margin=(0, 10)), widget,
                name=control.label, scroll=many, height=220 if many else None,
                sizing_mode="stretch_width",
            )
        elif control.kind == "toggle":
            widget = pn.widgets.Checkbox(label=control.label, value=bool(value))
            watched = "value"
        elif control.kind == "range":
            low, high, step = control.options
            widget = pn.widgets.IntSlider(
                label=control.label, start=int(low), end=int(high), step=int(step),
                value=int(low if value is None else value),
            )
            # A slider sweeps: redraw where it is released, not at every step.
            watched = "value_throttled"
        else:
            widget = pn.widgets.TextInput(label=control.label, value=value)
            watched = "value"
        widget.param.watch(setter(control.name), watched)
        widgets[control.name] = widget
        laid_out.append(widget if shown is None else shown)
        if control.group == "slice" and control.kind in ("choice", "range"):
            player, value_at, position_of = _player(pn, control, state[control.name])

            def play(event: Any, name: str = control.name, value_at: Callable[[int], Any] = value_at) -> None:
                if not busy["on"]:
                    apply_value(name, value_at(int(event.new)), from_player=True)

            player.param.watch(play, "value")
            players[control.name] = (player, value_at, position_of)
            interval = pn.widgets.IntInput(
                label="Frame interval [ms]", value=PLAY_INTERVAL_MS, start=50, step=50,
            )
            _link_interval(interval, player)
            laid_out.append(pn.Column(player, interval, name=f"Play: {control.label}", sizing_mode="stretch_width"))

    state.subscribe(follow)
    return laid_out


def _link_interval(box: Any, player: Any) -> None:
    """The interval box and the player's own slower/faster buttons, in step.

    An emptied box is ignored rather than handed to the player, which takes
    only a number.
    """
    def to_player(event: Any) -> None:
        if event.new is not None and event.new != player.interval:
            player.interval = int(event.new)

    def to_box(event: Any) -> None:
        if event.new != box.value:
            box.value = int(event.new)

    box.param.watch(to_player, "value")
    player.param.watch(to_box, "interval")


#: Milliseconds between frames when playing; each frame is a redraw on the
#: server, so the default leaves room for a Plotly rebuild.
PLAY_INTERVAL_MS = 700


def _player(pn: Any, control: Any, current: Any) -> tuple[Any, Callable[[int], Any], Callable[[Any], int | None]]:
    """A play/step/loop control over a slice or time control (#1359 video, step 1).

    It walks the same control the reader can set by hand -- the stored
    equilibrium slices, a camera's frames -- so playback is the plot drawn
    frame after frame, with every other control as chosen.
    """
    if control.kind == "choice":
        options = list(control.options)

        def value_at(position: int) -> Any:
            return options[max(0, min(position, len(options) - 1))]

        def position_of(value: Any) -> int | None:
            return options.index(value) if value in options else None

        start, end, step = 0, len(options) - 1, 1
    else:
        low, high, step = (int(v) for v in control.options)

        def value_at(position: int) -> Any:
            return max(low, min(position, high))

        def position_of(value: Any) -> int | None:
            return None if value is None else int(value)

        start, end = low, high
    position = position_of(current)
    player = pn.widgets.Player(
        label=f"Play: {control.label}", start=start, end=end, step=step,
        value=start if position is None else position, interval=PLAY_INTERVAL_MS,
        loop_policy="loop", show_value=False, scale_buttons=0.8, sizing_mode="stretch_width",
    )
    return player, value_at, position_of


__all__ = ["INDIVIDUAL_LABEL", "PLAY_INTERVAL_MS", "panel_controls"]
