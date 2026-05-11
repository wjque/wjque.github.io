async function fetchPosts() {
    const response = await fetch('./posts.json');
    if (!response.ok) {
        throw new Error('Failed to load posts index.');
    }

    const posts = await response.json();
    return posts.sort((a, b) => new Date(b.date) - new Date(a.date));
}

function escapeHtml(text) {
    return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function renderTags(tags) {
    return tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join('');
}

function formatDate(dateString) {
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) {
        return dateString;
    }

    return date.toLocaleDateString('en-CA', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
    });
}

function slugify(text) {
    return text
        .toLowerCase()
        .trim()
        .replace(/<[^>]+>/g, '')
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-');
}

function groupPostsBySection(posts) {
    return posts.reduce((groups, post) => {
        if (!groups[post.section]) {
            groups[post.section] = {
                title: post.sectionTitle,
                posts: []
            };
        }

        groups[post.section].posts.push(post);
        return groups;
    }, {});
}

function sortSectionEntries(sectionEntries) {
    return sectionEntries.sort(([leftKey], [rightKey]) => {
        if (leftKey === 'ideas' && rightKey !== 'ideas') {
            return 1;
        }

        if (leftKey !== 'ideas' && rightKey === 'ideas') {
            return -1;
        }

        return leftKey.localeCompare(rightKey);
    });
}

function setupIndexPage(posts) {
    const sectionList = document.getElementById('section-list');
    const postList = document.getElementById('blog-post-list');

    if (!sectionList || !postList) {
        return;
    }

    const sectionGroups = groupPostsBySection(posts);
    const sectionEntries = sortSectionEntries(Object.entries(sectionGroups));

    sectionList.innerHTML = sectionEntries.map(([sectionKey, sectionValue]) => {
        return `<a href="#section-${escapeHtml(sectionKey)}">${escapeHtml(sectionValue.title)}</a>`;
    }).join('');

    function renderSections() {
        const html = sectionEntries.map(([sectionKey, sectionValue]) => {
            const sectionLabel = sectionValue.title.toUpperCase();

            return `
                <section class="blog-section-group" id="section-${escapeHtml(sectionKey)}">
                    <div class="blog-section-header">
                        <div>
                            <p class="blog-section-kicker">${escapeHtml(sectionLabel)}</p>
                            <h3>${escapeHtml(sectionValue.title)}</h3>
                        </div>
                    </div>
                    <div class="blog-post-list">
                        ${sectionValue.posts.map((post) => `
                            <article class="blog-post-item">
                                <p class="blog-card-meta">${formatDate(post.date)} · ${escapeHtml(post.lang)}</p>
                                <h3><a href="post.html?id=${encodeURIComponent(post.id)}">${escapeHtml(post.title)}</a></h3>
                                <p class="blog-post-summary">${escapeHtml(post.summary)}</p>
                                <div class="blog-tags">${renderTags(post.tags)}</div>
                            </article>
                        `).join('')}
                    </div>
                </section>
            `;
        }).join('');

        postList.innerHTML = html;
    }

    renderSections();
}

function buildToc(container, tocContainer) {
    const headings = [...container.querySelectorAll('h2, h3')];

    if (headings.length === 0) {
        tocContainer.innerHTML = '<p class="blog-post-summary">No headings found.</p>';
        return;
    }

    tocContainer.innerHTML = headings.map((heading, index) => {
        const fallbackId = `section-${index}`;
        const generatedId = slugify(heading.textContent || fallbackId) || fallbackId;
        heading.id = generatedId;
        return `<a href="#${generatedId}" data-level="${heading.tagName === 'H2' ? '2' : '3'}">${escapeHtml(heading.textContent || '')}</a>`;
    }).join('');
}

async function setupPostPage(posts) {
    const titleEl = document.getElementById('post-title');
    const dateEl = document.getElementById('post-date');
    const langEl = document.getElementById('post-lang');
    const tagsEl = document.getElementById('post-tags');
    const contentEl = document.getElementById('post-content');
    const tocEl = document.getElementById('post-toc');

    if (!titleEl || !dateEl || !langEl || !tagsEl || !contentEl || !tocEl) {
        return;
    }

    const params = new URLSearchParams(window.location.search);
    const postId = params.get('id');
    const post = posts.find((item) => item.id === postId);

    if (!post) {
        document.title = 'Post Not Found | Wenjun Que';
        titleEl.textContent = 'Post not found';
        contentEl.innerHTML = '<p>Please return to the blog index and choose another post.</p>';
        tocEl.innerHTML = '<p class="blog-post-summary">Unavailable.</p>';
        return;
    }

    document.title = `${post.title} | Wenjun Que`;
    titleEl.textContent = post.title;
    dateEl.textContent = formatDate(post.date);
    langEl.textContent = post.lang;
    tagsEl.innerHTML = renderTags(post.tags);

    const markdownResponse = await fetch(`./${post.file}`);
    if (!markdownResponse.ok) {
        throw new Error('Failed to load post markdown.');
    }

    const markdown = await markdownResponse.text();
    contentEl.innerHTML = window.marked.parse(markdown, {
        breaks: false,
        gfm: true
    });

    if (window.renderMathInElement) {
        window.renderMathInElement(contentEl, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false }
            ],
            throwOnError: false
        });
    }

    buildToc(contentEl, tocEl);
}

async function initBlog() {
    try {
        const posts = await fetchPosts();
        setupIndexPage(posts);
        await setupPostPage(posts);
    } catch (error) {
        const titleEl = document.getElementById('post-title');
        const contentEl = document.getElementById('post-content');
        const postList = document.getElementById('blog-post-list');
        const message = error instanceof Error ? error.message : 'Unknown error';

        if (titleEl && contentEl) {
            titleEl.textContent = 'Unable to load post';
            contentEl.innerHTML = `<p>${escapeHtml(message)}</p>`;
        }

        if (postList) {
            postList.innerHTML = `<p class="blog-post-summary">${escapeHtml(message)}</p>`;
        }
    }
}

initBlog();
