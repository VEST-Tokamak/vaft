require(['gitbook', 'jquery'], function (gitbook, $) {
    var scrollFrame = null;

    function setPageTocOpen(open, restoreFocus) {
        var $book = $('.book');
        var $toggles = $('.js-page-toc-toggle');

        $book.toggleClass('with-page-toc', open);
        $toggles.attr('aria-expanded', open ? 'true' : 'false');

        if (open) {
            var toc = document.getElementById('page-toc');
            if (toc) {
                toc.focus({ preventScroll: true });
            }
        } else if (restoreFocus) {
            $('.page-toc-toggle').trigger('focus');
        }
    }

    $(document).on('click', '.js-page-toc-toggle', function (event) {
        event.preventDefault();
        var isOpen = $('.book').hasClass('with-page-toc');
        setPageTocOpen(!isOpen, isOpen);
    });

    function targetForHash(hash) {
        if (!hash || hash.charAt(0) !== '#') {
            return null;
        }
        try {
            return document.getElementById(decodeURIComponent(hash.slice(1)));
        } catch (error) {
            return null;
        }
    }

    function setActiveLink(hash) {
        $('.page-toc-item').removeClass('active');
        if (!hash) {
            return;
        }
        $('.page-toc-link').filter(function () {
            return this.hash === hash;
        }).first().closest('.page-toc-item').addClass('active');
    }

    function scrollArticleToHash(hash, updateHistory) {
        var target = targetForHash(hash);
        var inner = document.querySelector('.body-inner');
        if (!target || !inner) {
            return false;
        }
        var top = target.getBoundingClientRect().top - inner.getBoundingClientRect().top + inner.scrollTop - 16;
        // Assign scrollTop directly so the target is positioned before a mobile
        // drawer transition can trigger another browser-managed scroll.
        inner.scrollTop = Math.max(0, top);
        if (updateHistory && window.history && window.history.pushState) {
            window.history.pushState(null, '', hash);
        }
        setActiveLink(hash);
        return true;
    }

    function syncActiveLinkToScroll() {
        var inner = document.querySelector('.body-inner');
        if (!inner) {
            return;
        }
        var headings = $('.page-toc-link').map(function () {
            return targetForHash(this.hash);
        }).get().filter(Boolean);
        var threshold = inner.getBoundingClientRect().top + 72;
        var active = headings.length ? headings[0] : null;
        headings.some(function (heading) {
            if (heading.getBoundingClientRect().top <= threshold) {
                active = heading;
                return false;
            }
            return true;
        });
        setActiveLink(active ? '#' + active.id : '');
    }

    function bindArticleScroll() {
        $('.body-inner').off('scroll.vaftPageToc').on('scroll.vaftPageToc', function () {
            if (scrollFrame !== null) {
                return;
            }
            scrollFrame = window.requestAnimationFrame(function () {
                scrollFrame = null;
                syncActiveLinkToScroll();
            });
        });
    }

    $(document).on('click', '.page-toc-link', function (event) {
        if (scrollArticleToHash(this.hash, true)) {
            event.preventDefault();
        }
        if ($(document).width() <= 1240) {
            setPageTocOpen(false, false);
        }
    });

    $(document).on('keydown', function (event) {
        if (event.key === 'Escape' && $('.book').hasClass('with-page-toc')) {
            setPageTocOpen(false, true);
        }
    });

    // Generated reference pages: filter the function index (and, on a
    // category page, the entries below it) by name, category or summary.
    $(document).on('input', '[data-ref-filter]', function () {
        var needle = this.value.trim().toLowerCase();
        var $index = $(this).closest('[data-ref-index]');
        var visible = {};
        var shown = 0;
        $index.find('[data-ref-row]').each(function () {
            var row = $(this);
            var match = !needle ||
                (row.attr('data-ref-row') + ' ' + row.text()).toLowerCase().indexOf(needle) !== -1;
            row.prop('hidden', !match);
            if (match) {
                visible[row.attr('data-ref-row').split(' ')[0]] = true;
                shown += 1;
            }
        });
        $index.find('.ref-filter-empty').prop('hidden', shown > 0);
        $('.ref-entry').each(function () {
            $(this).prop('hidden', !visible[this.id]);
        });
    });

    $(window).on('hashchange', function () {
        scrollArticleToHash(window.location.hash, false);
    });

    gitbook.events.on('page.change', function () {
        setPageTocOpen(false, false);
        bindArticleScroll();
        window.setTimeout(function () {
            if (!scrollArticleToHash(window.location.hash, false)) {
                syncActiveLinkToScroll();
            }
        }, 0);
    });

    bindArticleScroll();
    window.setTimeout(function () {
        if (!scrollArticleToHash(window.location.hash, false)) {
            syncActiveLinkToScroll();
        }
    }, 0);
});
