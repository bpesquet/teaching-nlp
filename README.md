# Teaching - NLP

This repository contains the source code for my Natural Language Processing (NLP) introductory course.

Slideshows are written in [Typst](https://github.com/typst/typst) and rendered with [Touying](https://github.com/touying-typ/touying). Labs use [Googla Colaboratory](https://colab.research.google.com/).

## Table of Contents

### 📚 Lectures

- [From text to vectors](lectures/from_text_to_vectors/)
- [What happens when you prompt](lectures/what_happens_when_you_prompt/)

### 👩🏽‍💻 Labs

- [From text to vectors](https://colab.research.google.com/drive/1oCep0K_QvcY4MHIJDl2KttlSNehDj9Fh?usp=sharing)

## Development notes

```bash
# Generate the PDF version of a slideshow
typst compile lectures/{chapter}/{filename}.typ
```

I use the [Typst VS Code extension](https://marketplace.visualstudio.com/items?itemName=myriad-dreamin.tinymist) for showcasing slideshows during classes.

## License

[Creative Commons](LICENSE) for textual content and [MIT](CODE_LICENSE) for code.

Copyright © 2026-present [Baptiste Pesquet](https://bpesquet.fr).
