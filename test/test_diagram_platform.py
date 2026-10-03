"""The VEST data platform diagram (#1550): content from the declarative constants, layout as specified."""

import re

import pytest

import vaft.diagram
from vaft.diagram import _platform as P
from vaft.diagram._scene import Arrow, Label, Polyline


def _texts(scene):
    return " ".join(item.text for item in scene.items if isinstance(item, Label))


def _plain(text):
    """The LaTeX of a label reduced to its words: escapes and commands dropped."""
    text = re.sub(r"\\(textbf|ttfamily|small|footnotesize|color\{[^}]*\})", " ", text)
    return text.replace("\\&", "&").replace("\\_", "_").replace("\\{", "{").replace("\\}", "}")


@pytest.fixture(scope="module")
def platform():
    return vaft.diagram.vest_data_platform()


def test_every_declared_item_is_drawn(platform):
    text = _plain(_texts(platform.scene))
    names = (list(P.EXPERIMENTAL_PROCESSING) + list(P.RECONSTRUCTION) + list(P.DERIVED_PHYSICS)
             + list(P.ACCESS) + list(P.SIMULATION) + [P.DATABASE_ROOT, P.DATABASE_MASTER]
             + [n for group in P.DATABASE_LAYOUT for n in group])
    for domain, entries in P.SIMULATION.items():
        names += [concept for concept, _ in entries] + [implementation for _, implementation in entries]
    for name in names:
        assert name in text, name
    for value in P.EXECUTION_PLATFORMS + P.EXECUTION_BACKENDS:
        assert value in text


def test_the_issue_acceptance_content():
    assert P.EXPERIMENTAL_PROCESSING == ("Machine Model & History", "Signal Processing", "Quality & Validation",
                                         "Fault & Anomaly Detection", "Shot Classification", "Event Detection")
    assert P.RECONSTRUCTION == ("Eddy Current Model", "Magnetic EFIT", "Profile Fitting",
                                "Plasma Parameter Inference", "Kinetic EFIT")
    assert P.DERIVED_PHYSICS == ("Vacuum Field Proxies", "MHD Parameters", "Synthetic Diagnostics",
                                 "Coordinate Conversion", "Power Balance")
    assert list(P.SIMULATION) == ["Equilibrium", "Stability", "3D Response & Topology", "Transport"]
    assert [code for _, code in P.SIMULATION["3D Response & Topology"]] == ["GPEC", "FLARE"]
    assert P.ACCESS == ("Python API", "CLI", "GUI", "MCP", "Documentation")
    assert all(group[-1] == "..." for group in P.DATABASE_LAYOUT)  # representative, not exhaustive


def test_gpec_and_flare_sit_under_3d_response(platform):
    roles = {item.role for item in platform.scene.items}
    assert "simulation:3D Response & Topology:GPEC" in roles
    assert "simulation:3D Response & Topology:FLARE" in roles


def test_no_backend_names_in_either_figure(platform):
    for d in (platform, vaft.diagram.vest_data_platform_overview()):
        text = _texts(d.scene)
        for term in P.IMPLEMENTATION_TERMS:
            assert term not in text, term


def test_machine_outside_the_server_and_sections_inside(platform):
    x0, x1, y0, y1 = platform.model["server"]
    mx0, mx1, my0, my1 = platform.model["machine_extent"]
    assert mx1 < x0
    for name in ("processing", "database", "inference", "simulation"):
        b = platform.model["boxes"][name]
        assert x0 < b.x - b.width / 2 and b.x + b.width / 2 < x1, name
        assert y0 < b.y - b.height / 2 and b.y + b.height / 2 < y1, name
    access = platform.model["boxes"]["access"]
    assert access.y + access.height / 2 < y0  # client side, below the server
    ey0, ey1 = platform.model["execution"]
    assert ey1 < access.y - access.height / 2  # the foundation under everything


def test_only_the_data_relationships_are_drawn(platform):
    edges = platform.model["edges"]
    assert sorted(edges) == sorted([("machine", "processing", "forward"), ("processing", "database", "forward"),
                                    ("database", "inference", "both"), ("database", "simulation", "both"),
                                    ("database", "access", "forward")])
    arrows = [it for it in platform.scene.items if isinstance(it, Arrow)]
    assert len(arrows) == len(edges)
    both = {it.role for it in arrows if it.both}
    assert both == {"edge:database->inference", "edge:database->simulation"}


def test_the_database_is_the_central_hub(platform):
    boxes = platform.model["boxes"]
    db = boxes["database"]
    assert boxes["processing"].x < db.x < boxes["simulation"].x
    assert boxes["inference"].y > db.y > boxes["access"].y


def test_the_machine_is_the_packaged_render_embedded_in_the_svg(platform):
    from vaft.diagram._render import image_path, image_sha256
    from vaft.diagram._scene import Image

    images = [it for it in platform.scene.items if isinstance(it, Image)]
    assert [im.name for im in images] == [P.MACHINE_IMAGE]
    assert images[0].height / images[0].width == pytest.approx(P.MACHINE_ASPECT)
    assert image_path(P.MACHINE_IMAGE).is_file()
    assert f"image sha256={image_sha256(P.MACHINE_IMAGE)}" in platform.tikz  # a new picture is a new source


def test_simulation_entries_put_the_concept_first():
    eq = dict(P.SIMULATION["Equilibrium"])
    assert eq["Free Boundary"] == "TokaMaker" and eq["Fixed Boundary"] == "CHEASE"
    assert dict(P.SIMULATION["3D Response & Topology"]) == {"Plasma Response": "GPEC",
                                                            "Field-Line Following": "FLARE"}
    d = vaft.diagram.vest_data_platform()
    label = next(it for it in d.scene.items
                 if isinstance(it, Label) and it.role == "simulation:Equilibrium:TokaMaker")
    assert label.text.index("Free Boundary") < label.text.index("TokaMaker")


def test_the_overview_has_five_stages_and_no_file_contents():
    d = vaft.diagram.vest_data_platform_overview()
    assert len(d.model["stages"]) == 5
    assert ".h5" not in _texts(d.scene)
    roles = {it.role for it in d.scene.items}
    for stage in ("processing", "database", "reconstruction", "simulation", "access"):
        assert f"stage:{stage}" in roles
    assert ("database", "simulation", "both") in d.model["edges"]


@pytest.mark.parametrize("name", ["vest_data_platform", "vest_data_platform_overview"])
def test_deterministic_and_labels_off(name):
    fn = getattr(vaft.diagram, name)
    assert name in vaft.diagram.__all__
    assert fn().tikz == fn().tikz
    assert not [it for it in fn(labels=False).scene.items if isinstance(it, Label)]
    with pytest.raises(ValueError):
        fn(labels="yes")


def test_an_image_item_is_staged_and_inlined_without_external_references(tmp_path):
    from vaft.diagram import _render as R

    doc = R.tikz_document(vaft.diagram.vest_data_platform_overview().scene)
    names = R._stage_images(doc, tmp_path)
    assert names == [P.MACHINE_IMAGE] and (tmp_path / P.MACHINE_IMAGE).is_file()
    svg = R._embed_images(f"<image xlink:href='{P.MACHINE_IMAGE}'/>", names, tmp_path)
    assert "data:image/jpeg;base64," in svg and f"'{P.MACHINE_IMAGE}'" not in svg
    assert R._VAFTIMAGE_DVISVGM in R.template()  # the PDF export swaps exactly this definition


def test_the_database_technology_is_one_muted_caption(platform):
    captions = [it for it in platform.scene.items if isinstance(it, Label) and it.role == "database:technology"]
    assert len(captions) == 1 and captions[0].style == "concept annotation"
    assert captions[0].text == " · ".join(P.DATABASE_TECHNOLOGY)
    others = _texts(platform.scene).replace(captions[0].text, "")
    for term in P.DATABASE_TECHNOLOGY:
        assert term not in others.replace("IMAS ", "")  # named only in the caption


def test_the_access_caption_lists_data_access_first():
    assert P.ACCESS_CAPTION.startswith("Data Access")


def _jpeg_size(path):
    """(width, height) from a JPEG's start-of-frame marker."""
    data = path.read_bytes()
    i = 2
    while i < len(data):
        marker, length = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + length
    raise ValueError("no frame header")


def test_the_drawn_aspect_is_the_picture_aspect():
    from vaft.diagram._render import image_path

    w, h = _jpeg_size(image_path(P.MACHINE_IMAGE))
    assert P.MACHINE_ASPECT == pytest.approx(h / w)


def test_images_scale_with_a_transformed_scene():
    from vaft.diagram._scene import Image, Scene

    scene = Scene((Image((1.0, 1.0), P.MACHINE_IMAGE, 2.0, 3.0),)).transformed(scale=0.5, offset=(1.0, 0.0))
    (im,) = scene.items
    assert (im.at, im.width, im.height) == ((1.5, 0.5), 1.0, 1.5)
    for bad in ("../x.jpg", "x.gif", "sub/x.png"):
        with pytest.raises(ValueError):
            Image((0.0, 0.0), bad, 1.0, 1.0)


@pytest.mark.parametrize("quote", ["'", '"'])
def test_inlining_accepts_either_quote_and_refuses_to_leave_a_reference(tmp_path, quote):
    from vaft.diagram import _render as R

    (tmp_path / P.MACHINE_IMAGE).write_bytes(R.image_path(P.MACHINE_IMAGE).read_bytes())
    svg = R._embed_images(f"<image xlink:href={quote}{P.MACHINE_IMAGE}{quote}/>", [P.MACHINE_IMAGE], tmp_path)
    assert "data:image/jpeg;base64," in svg
    with pytest.raises(RuntimeError):
        R._embed_images(f"<image href=({P.MACHINE_IMAGE})/><x href='{P.MACHINE_IMAGE}?'/>", [P.MACHINE_IMAGE],
                        tmp_path)


@pytest.mark.skipif(not __import__("shutil").which("latex") or not __import__("shutil").which("dvisvgm"),
                    reason="latex and dvisvgm are not installed")
def test_the_rendered_svg_inlines_the_picture_and_references_nothing_outside():
    svg = vaft.diagram.vest_data_platform_overview().svg
    assert "data:image/jpeg;base64," in svg
    assert not re.search(r"""href=['"](?!data:|#)""", svg)
