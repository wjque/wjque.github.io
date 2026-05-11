# wjque.github.io

Personal website.

## Structure

- `index.html`: homepage
- `style.css`: shared site styles
- `images/`: homepage assets
- `blogs/index.html`: blog index
- `blogs/post.html`: blog entry page
- `blogs/blog.js`: blog rendering logic
- `blogs/blog.css`: blog styles
- `blogs/posts.json`: blog index data
- `blogs/posts/`: markdown source files

## Blog Sections

The blog is split by section.

- `blogs/posts/ideas/`: dated research thoughts and ongoing notes
- `blogs/posts/generative-models/`: diffusion, generation, and related model notes
- `blogs/posts/representation-learning/`: representation methods such as DINO, CLIP, or MAE
- `blogs/posts/training-systems/`: distributed training, optimization, and systems notes

New sections can be added later. The section order on the page follows the section name alphabetically.

## Adding a Post

1. Put a new markdown file under the right section in `blogs/posts/`.
2. Add one entry to `blogs/posts.json`.
3. Fill in these fields:
   - `id`
   - `section`
   - `sectionTitle`
   - `title`
   - `summary`
   - `date`
   - `lang`
   - `tags`
   - `file`

For `Ideas`, the file can keep the date in its filename, but the displayed title should stay clean and only use the actual topic.

For topic-based sections, the filename can simply follow the topic itself.

## Adding a New Section

1. Create a new directory under `blogs/posts/`.
2. Put markdown files into that directory.
3. Use the same section key in `blogs/posts.json`.
4. Choose a clear `sectionTitle` for display.

The page will pick it up automatically and place it by section name order.

## Updating the Site

- Homepage content lives mainly in `index.html`
- Navigation is shared manually across `index.html`, `blogs/index.html`, and `blogs/post.html`
- Blog list and section rendering are driven by `blogs/posts.json`
- Blog post content itself stays in markdown

## General Maintenance

- Keep page copy short
- Avoid decorative explainer text unless it adds real value
- Use English in the site UI
- Blog content can still be written in Chinese or mixed Chinese and English
