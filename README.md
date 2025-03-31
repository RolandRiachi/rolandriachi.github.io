# Math & ML Blog

A Jekyll-based blog for sharing educational content about mathematics and machine learning.

## Features

- Modern, responsive design
- Support for mathematical equations using MathJax
- Code syntax highlighting with Prism.js
- Category and tag support
- Social media integration
- Disqus comments (optional)

## Setup

1. Install Ruby and Jekyll:
   ```bash
   # On macOS
   brew install ruby
   gem install bundler jekyll
   ```

2. Install dependencies:
   ```bash
   bundle install
   ```

3. Run the development server:
   ```bash
   bundle exec jekyll serve
   ```

4. Visit `http://localhost:4000` to preview your site.

## Creating New Posts

Create new posts in the `_posts` directory with the following naming convention:
```
YYYY-MM-DD-title.md
```

Example front matter:
```yaml
---
layout: post
title: "Your Post Title"
date: YYYY-MM-DD
categories: [Category1, Category2]
tags: [tag1, tag2]
author: Your Name
---
```

## Deployment

This site is configured for GitHub Pages deployment. Simply push your changes to the main branch of your repository, and GitHub Pages will automatically build and deploy your site.

## Customization

- Edit `_config.yml` to modify site settings
- Modify `assets/css/main.css` to customize the site's appearance
- Add new layouts in the `_layouts` directory
- Create new includes in the `_includes` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details. 