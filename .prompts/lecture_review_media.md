# Lecture Media & Visual Review Prompt

Review the lecture for **visual/media coverage** against AGENTS.md requirements:

## Check Against AGENTS.md

1. **Visual requirements** (from AGENTS.md):
   > "each section/subsection should include (1) brief prose intro/explanation, (2) optional visual or `#FIXME` placeholder..."

   - Does EVERY subsection (###) have a visual or `#FIXME` placeholder?
   - List any subsections missing visuals

2. **Humor distribution** (from AGENTS.md):
   > "Sprinkle humor/comics throughout (use existing assets—never invent links)"

   - Are XKCD comics distributed throughout the lecture?
   - Is humor frontloaded or evenly spread across all major sections?
   - Count XKCD comics per major section (Quick hits, Defensive, Debugging)

3. **Image sourcing** (from AGENTS.md):
   > "prefer images local to the lecture folder; if reusing from elsewhere, copy into the lecture's `media/` subdir first"
   > "`all_xkcd.html` lists available XKCD panels—pick from there and copy locally instead of hotlinking"

   - Are all images referenced from local `media/` folder?
   - Are there any external/hotlinked images?

4. **URL format** (from AGENTS.md):
   > "GitHub Pages base URL: `https://christopherseaman.github.io/datasci_223/`"

   - Check image paths: should be `/01/media/...` for GitHub Pages deployment
   - Are all image paths in the correct format?

## Media Folder Audit

1. **List the lecture's `media/` folder contents**
2. **Compare** to images referenced in the markdown
3. **Identify unused assets**: Images in `media/` not referenced in lecture
4. **Suggest placements**: For each unused asset, suggest where it could fit thematically

## Output Format

- **Visual coverage**: List subsections with/without visuals
- **Humor distribution**: Count by section, note if uneven
- **URL format**: List any incorrect image paths
- **Unused assets**: List with suggested placements
