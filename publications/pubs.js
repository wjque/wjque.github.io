(function() {
    var homeEl = document.getElementById('home-pubs');
    var fullEl = document.getElementById('full-pubs');

    var isHome = !!homeEl;
    var jsonPath = isHome ? 'publications/pubs.json' : 'pubs.json';

    if (!homeEl && !fullEl) return;

    fetch(jsonPath)
        .then(function(res) {
            if (!res.ok) throw new Error('Failed to load publications data');
            return res.json();
        })
        .then(function(pubs) {
            pubs.sort(function(a, b) { return b.year - a.year; });
            if (isHome) {
                renderHome(pubs.slice(0, 3), homeEl);
            } else {
                renderFull(pubs, fullEl);
            }
        })
        .catch(function(err) {
            var el = homeEl || fullEl;
            if (el) el.innerHTML = '<p style="color:var(--text-light);text-align:center">Unable to load publications.</p>';
        });

    function renderHome(pubs, container) {
        container.innerHTML = pubs.map(function(pub) {
            var imgTag = pub.image
                ? '<img src="images/publications/' + esc(pub.image) + '" alt="Paper thumbnail">'
                : '';
            return '<div class="publication-item">' +
                '<div class="publication-img">' + imgTag + '</div>' +
                '<div class="publication-content">' +
                    '<h3>' + esc(pub.title) + '</h3>' +
                    '<p class="authors">' + renderAuthors(pub.authors) + '</p>' +
                    (pub.footnote ? '<p class="footnote">' + esc(pub.footnote) + '</p>' : '') +
                    '<p class="venue">' + esc(pub.venue) + '</p>' +
                    renderLinks(pub.links) +
                '</div>' +
            '</div>';
        }).join('');
    }

    function renderFull(pubs, container) {
        container.innerHTML = pubs.map(function(pub) {
            var imgTag = pub.image
                ? '<img src="../images/publications/' + esc(pub.image) + '" alt="Paper thumbnail">'
                : '';
            return '<div class="publication-item pub-full">' +
                '<div class="publication-img">' + imgTag + '</div>' +
                '<div class="publication-content">' +
                    '<h3>' + esc(pub.title) + '</h3>' +
                    '<p class="authors">' + renderAuthors(pub.authors) + '</p>' +
                    (pub.footnote ? '<p class="footnote">' + esc(pub.footnote) + '</p>' : '') +
                    '<p class="venue">' + esc(pub.venue) + '</p>' +
                    renderLinks(pub.links) +
                '</div>' +
            '</div>';
        }).join('');
    }

    function renderAuthors(authors) {
        return authors.map(function(a) {
            var name = esc(a.name);
            if (a.highlight) {
                if (a.sup) {
                    return '<span class="author-highlight">' + name + '<sup class="' + (a.supClass || '') + '">' + esc(a.sup) + '</sup></span>';
                }
                return '<span class="author-highlight">' + name + '</span>';
            }
            if (a.sup) {
                return name + '<sup class="' + (a.supClass || '') + '">' + esc(a.sup) + '</sup>';
            }
            return name;
        }).join(', ');
    }

    function renderLinks(links) {
        if (!links) return '';
        var parts = [];
        if (links.pdf) parts.push('<a href="' + esc(links.pdf) + '" class="btn">PDF</a>');
        if (links.code) parts.push('<a href="' + esc(links.code) + '" class="btn">Code</a>');
        if (links.cite) parts.push('<a href="' + esc(links.cite) + '" class="btn">Cite</a>');
        return parts.length ? '<div class="publication-links">' + parts.join('') + '</div>' : '';
    }

    function esc(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }
})();
