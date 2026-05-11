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
- `blogs/posts/`: markdown source files

## News

The `News` section is maintained directly in `index.html` under `#news`.

Rules:

1. Edit the `<ul class="news-list">` entries directly.
2. Use `YYYY.MM` in `<span class="news-date">`.
3. Keep each summary short.
4. Put newer items first.

## Publications

The `Publications` section is maintained directly in `index.html` under `#publications`.

Rules:

1. Duplicate or edit one `.publication-item` block.
2. Fill in title, authors, venue, and links.
3. Uncomment blocks if needed.
4. Keep the newest or most important items near the top.

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
