require(['gitbook', 'jquery'], function (gitbook, $) {
    function fallbackCopy(text) {
        var textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();

        try {
            return document.execCommand('copy');
        } catch (error) {
            console.warn('Copy to clipboard failed.', error);
            return false;
        } finally {
            document.body.removeChild(textarea);
        }
    }

    function copyText(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(text)
                .then(function () { return true; })
                .catch(function () { return fallbackCopy(text); });
        }

        return Promise.resolve(fallbackCopy(text));
    }

    function addCopyButtons() {
        $('.markdown-section pre').each(function () {
            var $pre = $(this);
            var isMermaid = $pre.hasClass('mermaid-diagram') ||
                $pre.find('.language-mermaid').length > 0;

            if (isMermaid || $pre.closest('.code-block-wrapper').length) {
                return;
            }

            var $highlight = $pre.parent('.highlight');
            var $block = $highlight.length ? $highlight : $pre;
            $block.wrap('<div class="code-block-wrapper"></div>');

            var $wrapper = $block.parent('.code-block-wrapper');
            var $button = $('<button>', {
                'type': 'button',
                'class': 'copy-code-button',
                'aria-label': 'Copy code to clipboard',
                'text': 'Copy'
            });

            $button.on('click', function () {
                var code = $wrapper.find('pre code').first().text();
                var button = this;

                copyText(code).then(function (copied) {
                    $(button).text(copied ? 'Copied' : 'Unable to copy');
                    window.setTimeout(function () {
                        $(button).text('Copy');
                    }, 2000);
                });
            });

            $wrapper.append($button);
        });
    }

    gitbook.events.on('start', addCopyButtons);
    gitbook.events.on('page.change', addCopyButtons);
    $(addCopyButtons);
});
