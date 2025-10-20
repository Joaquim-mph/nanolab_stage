# Pipeline Visualization Guide

**Professional visual documentation of the Nanolab data pipeline architecture**

This document explains how to generate and use pipeline visualizations created with Graphviz.

---

## Quick Start

```bash
# Install dependencies
pip install graphviz

# Install Graphviz system package
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz

# Generate all diagrams
python scripts/visualize_pipeline.py

# View generated diagrams
ls docs/images/
```

Output: Professional PNG diagrams in `docs/images/`

---

## Available Diagrams

The visualization script generates 5 professional diagrams:

### 1. Pipeline Overview (`pipeline_overview.png`)

**Purpose:** High-level 4-layer architecture overview

**Shows:**
- Layer 1: Raw CSV files
- Layer 2: Staged Parquet (Bronze)
- Layer 3: Intermediate preprocessing (Silver)
- Layer 4: Analysis results (Gold)
- Layer 5: Visualization output
- Data flow between layers
- Key characteristics (idempotent, parallel, validated)

**Use for:**
- README documentation
- Presentation slides
- Architecture discussions
- Onboarding new team members

**Example:**
```bash
python scripts/visualize_pipeline.py --diagram overview --format png --format svg
```

### 2. Detailed Data Flow (`pipeline_detailed.png`)

**Purpose:** Detailed data transformations and file formats

**Shows:**
- Specific file formats (CSV → Parquet)
- Partitioning schemes (proc/date/run_id)
- Column transformations
- Schema evolution through layers
- Metadata enrichment
- Processing components

**Use for:**
- Technical documentation
- Debugging data issues
- Understanding transformations
- Schema design discussions

**Example:**
```bash
python scripts/visualize_pipeline.py --diagram detailed --format pdf
```

### 3. CLI Command Flow (`pipeline_cli.png`)

**Purpose:** CLI architecture and command relationships

**Shows:**
- Entry point: `nanolab-pipeline.py`
- Available commands (pipeline, stage, preprocess)
- Default paths configuration
- Command-line options
- Subprocess isolation pattern
- Configuration flow

**Use for:**
- CLI documentation
- User guides
- Command reference
- Understanding subprocess architecture

**Example:**
```bash
python scripts/visualize_pipeline.py --diagram cli
```

### 4. IV Analysis Workflow (`pipeline_iv_analysis.png`)

**Purpose:** Complete IV measurement analysis pipeline

**Shows:**
- Analysis scripts (Layer 4)
  - `aggregate_iv_stats.py`
  - `compute_hysteresis.py`
  - `analyze_hysteresis_peaks.py`
- Data dependencies
- Intermediate outputs
- Visualization scripts
- Publication figure generation
- Execution order

**Use for:**
- Analysis workflow documentation
- Script dependencies
- Output file structure
- Planning new analyses

**Example:**
```bash
python scripts/visualize_pipeline.py --diagram iv_analysis --format svg
```

### 5. Procedure Template (`pipeline_procedure_template.png`)

**Purpose:** Steps to add new measurement procedures

**Shows:**
- Step 1: Define YAML schema
- Step 2: Create Pydantic models
- Step 3: Implement preprocessing
- Step 4: Implement analysis
- Step 5: Add visualization
- Step 6: Integrate with CLI
- Step 7: Test & document
- Reference to `ADDING_NEW_PROCEDURES.md`

**Use for:**
- Extending the pipeline
- Developer onboarding
- Planning new procedures (IVg, ITt, etc.)
- Process documentation

**Example:**
```bash
python scripts/visualize_pipeline.py --diagram procedure_template
```

---

## Usage Examples

### Generate All Diagrams (Default)

```bash
# PNG format, 300 DPI
python scripts/visualize_pipeline.py
```

### Generate Specific Diagram

```bash
# Just the overview
python scripts/visualize_pipeline.py --diagram overview

# Just the CLI flow
python scripts/visualize_pipeline.py --diagram cli

# Just IV analysis workflow
python scripts/visualize_pipeline.py --diagram iv_analysis
```

### Multiple Formats

```bash
# PNG and SVG
python scripts/visualize_pipeline.py --format png --format svg

# PNG, SVG, and PDF
python scripts/visualize_pipeline.py --diagram overview --format png --format svg --format pdf

# SVG only (vector, scales perfectly)
python scripts/visualize_pipeline.py --format svg
```

### Custom Output Directory

```bash
# Save to custom location
python scripts/visualize_pipeline.py --output-dir ./my_diagrams

# Save to project root
python scripts/visualize_pipeline.py --output-dir .
```

### High-Resolution Diagrams

```bash
# 600 DPI for publication
python scripts/visualize_pipeline.py --dpi 600 --format png

# Vector format (infinite resolution)
python scripts/visualize_pipeline.py --format svg
python scripts/visualize_pipeline.py --format pdf
```

### Complete Example

```bash
# Generate all diagrams in multiple formats with high DPI
python scripts/visualize_pipeline.py \
  --diagram all \
  --format png \
  --format svg \
  --format pdf \
  --dpi 600 \
  --output-dir docs/images
```

---

## Format Guide

### PNG (Raster)

**Pros:**
- Universal compatibility
- Easy to embed in documents
- Good for web/presentations

**Cons:**
- Fixed resolution (pixelated when scaled)
- Larger file sizes at high DPI

**Recommended for:**
- README files
- GitHub documentation
- Quick viewing

**Command:**
```bash
python scripts/visualize_pipeline.py --format png --dpi 300
```

### SVG (Vector)

**Pros:**
- Infinite scalability (no pixelation)
- Smaller file sizes
- Editable in Inkscape/Illustrator
- Perfect for web

**Cons:**
- Some tools don't support SVG well

**Recommended for:**
- Web documentation
- Interactive diagrams
- Presentations (when supported)
- Editing/customization

**Command:**
```bash
python scripts/visualize_pipeline.py --format svg
```

### PDF (Vector)

**Pros:**
- Vector format (scalable)
- Universal compatibility
- Good for print
- Preserves quality

**Cons:**
- Larger file sizes than SVG
- Less web-friendly

**Recommended for:**
- Print documentation
- Technical reports
- Archival purposes
- High-quality handouts

**Command:**
```bash
python scripts/visualize_pipeline.py --format pdf
```

### DOT (Source)

**Pros:**
- Plain text source
- Version control friendly
- Editable
- Regeneratable

**Cons:**
- Not directly viewable
- Requires Graphviz to render

**Recommended for:**
- Version control
- Customization
- Advanced editing

**Command:**
```bash
python scripts/visualize_pipeline.py --format dot
```

---

## Using Diagrams in Documentation

### Markdown (README, GitHub)

```markdown
# Pipeline Architecture

![Pipeline Overview](docs/images/pipeline_overview.png)

## CLI Commands

![CLI Flow](docs/images/pipeline_cli.png)

## IV Analysis Workflow

![IV Analysis](docs/images/pipeline_iv_analysis.png)
```

### HTML

```html
<img src="docs/images/pipeline_overview.svg" alt="Pipeline Overview" width="800">
```

### LaTeX

```latex
\begin{figure}
  \centering
  \includegraphics[width=\textwidth]{docs/images/pipeline_overview.pdf}
  \caption{Nanolab Pipeline 4-Layer Architecture}
  \label{fig:pipeline}
\end{figure}
```

### reStructuredText (Sphinx)

```rst
.. figure:: docs/images/pipeline_overview.png
   :width: 800px
   :align: center
   :alt: Pipeline Overview

   Nanolab Pipeline 4-Layer Architecture
```

---

## Customization

The visualization script is designed to be easily customizable.

### Color Scheme

Edit the `COLORS` dictionary in `scripts/visualize_pipeline.py`:

```python
COLORS = {
    "layer1": "#E8F4F8",  # Light blue - Raw data
    "layer2": "#B8E6F0",  # Medium blue - Staged
    "layer3": "#88D8E8",  # Darker blue - Intermediate
    "layer4": "#58CAE0",  # Deep blue - Analysis
    "layer5": "#28BCD8",  # Deepest blue - Visualization
    "cli": "#FFE0B2",     # Light orange - CLI
    "script": "#FFF9C4",  # Light yellow - Scripts
    "config": "#E1BEE7",  # Light purple - Config
    "edge": "#546E7A",    # Blue-grey - Arrows
}
```

Colors are chosen to be:
- Professional
- Colorblind-friendly
- Print-friendly (grayscale compatible)
- Visually distinct

### Layout

Each diagram function supports different Graphviz layouts:

```python
dot.attr(
    rankdir='TB',     # TB = top-to-bottom, LR = left-to-right
    splines='ortho',  # ortho = orthogonal, polyline = angled
    nodesep='0.8',    # Horizontal spacing
    ranksep='1.2',    # Vertical spacing
)
```

### Add New Diagram

Create a new function in `scripts/visualize_pipeline.py`:

```python
def create_my_custom_diagram(output_dir: Path, formats: List[str]):
    """
    Create custom diagram for specific use case.
    """
    dot = graphviz.Digraph(name='my_custom_diagram')

    # Configure graph
    dot.attr(rankdir='TB', splines='ortho')

    # Add nodes
    dot.node('node1', 'My Node', fillcolor='#E8F4F8')
    dot.node('node2', 'Another Node', fillcolor='#B8E6F0')

    # Add edges
    dot.edge('node1', 'node2', label='Flow')

    # Render
    for fmt in formats:
        output_path = output_dir / f'my_custom_diagram.{fmt}'
        dot.format = fmt
        dot.render(str(output_path.with_suffix('')), cleanup=True)
        print(f"Created: {output_path}")
```

Register it in `main()`:

```python
diagrams_to_create = {
    'overview': create_overview_diagram,
    'detailed': create_detailed_diagram,
    'cli': create_cli_diagram,
    'iv_analysis': create_iv_analysis_diagram,
    'procedure_template': create_procedure_template_diagram,
    'my_custom': create_my_custom_diagram,  # Add here
}
```

---

## Troubleshooting

### "graphviz package not installed"

**Solution:**
```bash
pip install graphviz
```

### "Error: command 'dot' not found"

The Python package is installed, but the system Graphviz is missing.

**Solution:**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install graphviz
```

**macOS:**
```bash
brew install graphviz
```

**Windows:**
1. Download from https://graphviz.org/download/
2. Install and add to PATH
3. Restart terminal

**Verify installation:**
```bash
dot -V
```

Should output: `dot - graphviz version X.X.X`

### "Permission denied"

**Solution:**
```bash
chmod +x scripts/visualize_pipeline.py
```

### Low-quality PNG output

**Solution:** Increase DPI
```bash
python scripts/visualize_pipeline.py --dpi 600
```

Or use vector format:
```bash
python scripts/visualize_pipeline.py --format svg
```

### Diagrams too large/small

**Edit the script node attributes:**

```python
dot.attr('node',
    width='3.5',   # Increase/decrease node width
    height='0.8',  # Increase/decrease node height
    fontsize='11', # Increase/decrease font size
)
```

### Text overlapping

**Increase spacing:**

```python
dot.attr(
    nodesep='1.0',  # Increase horizontal spacing
    ranksep='1.5',  # Increase vertical spacing
)
```

---

## Advanced Usage

### Graphviz Attributes Reference

Full Graphviz documentation: https://graphviz.org/documentation/

**Common node shapes:**
- `box` - Rectangle
- `ellipse` - Oval
- `circle` - Circle
- `diamond` - Diamond
- `cylinder` - Database/storage
- `folder` - Directory
- `note` - Document with folded corner
- `component` - Component/module

**Common edge styles:**
- `solid` - Solid line (default)
- `dashed` - Dashed line
- `dotted` - Dotted line
- `bold` - Thick line

**Layout engines:**
- `dot` - Hierarchical (default)
- `neato` - Spring model
- `fdp` - Force-directed
- `circo` - Circular
- `twopi` - Radial

Change engine:
```python
dot = graphviz.Digraph(engine='neato')
```

### Programmatic Generation

Generate diagrams from code analysis:

```python
from pathlib import Path
import graphviz

def visualize_directory_structure(root: Path):
    """Generate diagram from actual directory structure"""
    dot = graphviz.Digraph(name='actual_structure')

    # Scan directories
    for path in root.rglob('*'):
        if path.is_dir():
            dot.node(str(path), path.name, shape='folder')
            if path.parent != root:
                dot.edge(str(path.parent), str(path))

    dot.render('structure', format='png', cleanup=True)

visualize_directory_structure(Path('data'))
```

### Integration with CI/CD

Add to `.github/workflows/docs.yml`:

```yaml
name: Generate Documentation

on: [push]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Graphviz
        run: sudo apt-get install -y graphviz

      - name: Install Python dependencies
        run: pip install graphviz

      - name: Generate diagrams
        run: python scripts/visualize_pipeline.py --output-dir docs/images

      - name: Commit diagrams
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add docs/images/*.png
          git commit -m "Update pipeline diagrams" || echo "No changes"
          git push
```

---

## Best Practices

### When to Regenerate Diagrams

Regenerate diagrams when:
- Architecture changes (new layers, new components)
- CLI commands change
- Adding new procedures
- Directory structure changes
- Data flow changes
- Before documentation releases

### Versioning

Keep diagram sources in version control:

```bash
# Generate and commit
python scripts/visualize_pipeline.py --format svg --format png
git add docs/images/*.svg docs/images/*.png
git commit -m "Update pipeline diagrams for v2.0"
```

### Documentation Standards

**In README.md:**
- Use PNG for quick loading
- Include alt text for accessibility
- Size appropriately (800px width max)

**In technical docs:**
- Use SVG for scalability
- Link to high-resolution versions
- Provide captions explaining context

**For presentations:**
- Use SVG or high-DPI PNG (600+)
- Use consistent styling across slides
- Add speaker notes explaining diagram

**For publications:**
- Use PDF for LaTeX
- Use high-DPI PNG (300+) for Word
- Verify grayscale readability
- Follow journal figure guidelines

---

## Related Documentation

- **CLAUDE.md** - Complete project overview
- **CLI_ARCHITECTURE.md** - CLI technical details
- **4LAYER_COMPLETE.md** - Architecture deep dive
- **ADDING_NEW_PROCEDURES.md** - Extension guide
- **QUICK_START.md** - Getting started guide

---

## Examples Gallery

After running the visualization script, you'll have:

```
docs/images/
├── pipeline_overview.png          # High-level 4-layer view
├── pipeline_detailed.png          # Detailed transformations
├── pipeline_cli.png               # CLI command flow
├── pipeline_iv_analysis.png       # IV workflow
└── pipeline_procedure_template.png # Extension guide
```

**Typical sizes (300 DPI PNG):**
- Overview: ~200 KB
- Detailed: ~300 KB
- CLI: ~150 KB
- IV Analysis: ~250 KB
- Procedure Template: ~180 KB

**Use these diagrams to:**
- Document your pipeline in README
- Create onboarding materials
- Design presentations
- Plan architecture changes
- Communicate with collaborators
- Write technical reports

---

## Support

For issues or questions:
1. Check this documentation
2. Review `scripts/visualize_pipeline.py` source
3. Consult Graphviz documentation: https://graphviz.org/
4. Open an issue in the project repository

---

**Generated diagrams are professional-quality and ready for:**
- GitHub documentation
- Technical presentations
- Academic publications
- Team onboarding
- Architecture reviews
- Grant proposals

Enjoy your beautiful pipeline visualizations! 🎨
