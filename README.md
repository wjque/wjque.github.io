# wjque.github.io

Personal website.

## Structure

- `index.html`: homepage
- `style.css`: shared site styles
- `images/`: homepage assets
- `assets/fonts/`: locally hosted font files
- `blogs/index.html`: blog index
- `blogs/post.html`: blog entry page
- `blogs/blog.js`: blog rendering logic
- `blogs/blog.css`: blog styles
- `blogs/posts.json`: blog index data
- `publications/index.html`: full publication list
- `publications/pubs.json`: central publication data (single source of truth)
- `publications/pubs.js`: publication renderer (consumed by both homepage and full page)
- `images/publications/`: paper thumbnails

## News

The `News` section is maintained directly in `index.html` under `#news`.

Rules:

1. Edit the `<ul class="news-list">` entries directly.
2. Use `YYYY.MM` in `<span class="news-date">`.
3. Keep each summary short.
4. Put newer items first.

## Publications

Publications are driven by a single JSON file: **`publications/pubs.json`**. Both the homepage card and the full publication page read from it — there is only one place to edit.

### pubs.json format

```json
{
    "title": "Paper Title",
    "authors": [
        { "name": "Wenjun Que", "highlight": true },
        { "name": "Other Author", "sup": "‡" },
        { "name": "Third Author", "sup": "†", "supClass": "project-leader" }
    ],
    "footnote": "‡ Corresponding author",
    "venue": "CVPR, 2024",
    "year": 2024,
    "image": "paper-title.png",
    "links": { "pdf": "https://...", "code": "https://...", "cite": "https://..." }
}
```

Field notes:
- `highlight: true` wraps the author name in `.author-highlight`.
- `sup` adds a superscript marker (e.g. `‡`, `*`, `†`); use `supClass` for special styling (e.g. `"project-leader"`).
- `footnote` is optional — omit or use `""` when not needed.
- `venue` should be a concise abbreviation (e.g. `CVPR, 2024`).
- `year` controls sort order (newest first). On the homepage, only the 3 most recent entries are shown.
- `image` points to a file under `images/publications/` — shown on both the homepage and the full page.
- `links` — omit any key to hide the corresponding button.

### Adding a publication

1. Append a new entry to the array in `publications/pubs.json`.
2. Place the thumbnail image in `images/publications/`.
3. Done — both the homepage and `/publications/` update automatically.

## Blogs

### Blog Sections

- `blogs/posts/ideas/`: dated research thoughts and ongoing notes
- `blogs/posts/generative-models/`: diffusion, generation, and related model notes
- `blogs/posts/representation-learning/`: representation methods such as DINO, CLIP, or MAE
- `blogs/posts/training-systems/`: distributed training, optimization, and systems notes

### Adding a Post

1. Put a new markdown file under the right section in `blogs/posts/`.
2. Add one entry to `blogs/posts.json`.
3. Fill in these fields:
   - `id`, `section`, `sectionTitle`
   - `title`, `summary`, `date`
   - `lang`, `tags`, `file`

For `Ideas`, the file can keep the date in its filename, but the displayed title should stay clean and only use the actual topic.

For topic-based sections, the filename can simply follow the topic itself.

### Adding a New Section

1. Create a new directory under `blogs/posts/`.
2. Put markdown files into that directory.
3. Use the same section key in `blogs/posts.json`.
4. Choose a clear `sectionTitle` for display.

### Blog Maintenance Notes

- Post list rendering is driven by `blogs/posts.json`
- Post body content stays in markdown
- Blog styles live in `blogs/blog.css`
- Blog math rendering is handled in `blogs/blog.js`
- For multiline math, prefer `$$ ... \begin{aligned} ... \end{aligned} ... $$`
- Keep short formulas inline when possible

## General Maintenance

- Keep page copy short
- Avoid decorative explainer text unless it adds real value
- Use English in the site UI
- Blog content can still be written in Chinese or mixed Chinese and English
