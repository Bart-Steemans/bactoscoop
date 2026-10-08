"""Generate source-backed reference pages and adapted notebook tutorials."""
import ast
import base64
import html
import json
import re
import shutil
import textwrap
from pathlib import Path
import tomllib
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
from docs_config import PACKAGE
DATA = json.loads((ROOT / 'reports/inspection.json').read_text(encoding='utf-8'))

def write(slug, text):
    p = SOURCE / (slug + '.rst')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join(line.rstrip() for line in text.strip().splitlines()) + '\n', encoding='utf-8')

def heading(title, char='='):
    return f'{title}\n{char * len(title)}\n\n'

def plain(value):
    """Escape spreadsheet prose as text rather than reStructuredText markup."""
    return re.sub(r'([*|_`])', r'\\\1', value)

def field_description(value):
    return value.replace('feature_names_in_','``feature_names_in_``').replace('Metadata_','``Metadata_``')

def documented_parameter(doc, name):
    """Recover parameter prose without copying stale argument lists."""
    lines = doc.splitlines()
    for index,line in enumerate(lines):
        if re.match(r'^\s*'+re.escape(name)+r'\s*(?:\([^)]*\))?\s*:',line):
            description = line.split(':',1)[1].strip()
            # NumPy's declaration line contains a type, followed by indented prose.
            if index+1 < len(lines) and lines[index+1].startswith('    '):
                paragraphs = []
                for following in lines[index+1:]:
                    if following and not following.startswith('    '):
                        break
                    if following.strip():
                        paragraphs.append(following.strip())
                if paragraphs:
                    description = ' '.join(paragraphs)
            return plain(description)
    return None

def code(value, language='python'):
    return f'.. code-block:: {language}\n\n' + textwrap.indent(value.rstrip(), '   ') + '\n\n'

PARAMETERS = {
'image_folder_path': 'Dataset folder containing TIFFs and the masks subfolder; None delays path configuration.',
'px': 'Micrometres per pixel. Geometry arrays remain in pixel coordinates.',
'error_policy': '"continue" records independent item/cell failures; "raise" raises the recorded error. Setup failures can still stop work.',
'log_cell_errors': 'Print each recorded cell error when true; recording is independent of this flag.',
'phase_channel': 'TIFF suffix identifying the phase images, such as "C1". An empty suffix is the signature default.',
'channels': 'Requested channel identifiers; for correlation, channel pairs are generated from this sequence.',
'channel_list': 'Sequence of channel suffixes to load or attach.',
'channel': 'Channel identifier for the selected signal, or None for phase/morphology where supported.',
'channel_method_pairs': 'Pairs of (channel list, feature method), for example [([None], "morphological"), (["C2"], "profiling")].',
'all_data': 'Feature-method-specific option; objects includes detailed per-object arrays when true.',
'reset': 'Clear stored feature dataframes when true; false retains other pairs while replacing each recalculated pair.',
'reset_channels': 'Clear loaded channel dictionaries and background/interpolation caches before loading requested channels.',
'shift_signal': 'Enable the method-specific optional signal cropping/shift path; this is not a universal object-coordinate shift.',
'use_shifted_contours': 'Select stored shifted contour, midline, and mesh geometry; the correction must already exist.',
'max_mesh_size': 'Maximum accepted mesh row count; cells above this limit are removed from the image cell population.',
'retain_contour_on_object_mesh_failure_channels': 'Channels whose object rows keep contour measurements and NaN mesh aggregates when object geometry fails.',
'retain_contour_on_object_mesh_failure': 'Per-cell objects option used by the collection retention-channel setting.',
'join_thresh': 'Maximum pole-to-pole joining distance in pixels.',
'split_thresh': 'Relative constriction threshold used by the splitting pipeline.',
'CD_width': 'Use the width-based constriction path when true; otherwise use the pipeline phase-signal path.',
'smoothing': 'Smoothing passed to the contour/mesh fitting path.',
'neighbor_filter_max_neighbors': 'Maximum number of touching distinct labels allowed before meshing; None disables the prefilter.',
'neighbor_filter_connectivity': '1 or 4 for edge contact; 2 or 8 for edge and diagonal contact.',
'mask_thresh': 'Threshold passed as Omnipose mask_threshold.',
'minsize': 'Minimum accepted segmentation-mask size in pixels.',
'n': 'Selected image positions; wrapper accepts None for all. Indices must be nonempty, unique valid integers.',
'model_name': 'Omnipose model identifier; default collection model is bact_phase_omni.',
'path_to_model': 'Path to a serialized SVM carrying the feature_names_in_ training-column schema.',
'cols': 'Legacy curation argument retained for compatibility; unused by the current curation preparation.',
'control': 'Request visual examples of retained/rejected classes; random selection can be unavailable for small classes.',
'save_data': 'Write the combined mesh table when true.',
'save_curated_data': 'Write curated mesh pickle when true; in-memory curation always occurs.',
'pkl_name': 'Mesh input filename or pickle output name. For feature output, .pkl means exact filename; a suffixless name is a legacy tag.',
'pkl_path': 'Directory containing an input mesh file; None selects the image folder.',
'parquet_name': 'Output filename ending in .parquet within the image folder; None chooses the dataset-based default.',
'include_metadata_tag': 'Rename identity columns with Metadata_ prefixes.',
'discard_morphological_nan': 'Drop merged rows missing cell_area or cell_length; these columns must exist.',
'output_subfolder': 'Destination for newly rasterized curated masks, relative to the dataset folder.',
'overwrite': 'Replace existing curated mask TIFFs when true; false raises for a file already present.',
'align': 'Use the object-detection alignment path for the selected signal.',
'log_sigma': 'Laplacian-of-Gaussian sigma for signal-object detection.',
'kernel_width': 'Width of the square dilation kernel used for candidate object masks.',
'min_overlap_ratio': 'Minimum accepted overlap ratio under the utility implementation; inspect its denominator in source.',
'max_external_ratio': 'Maximum accepted outside-cell ratio under the utility implementation.',
'object_list': 'Explicit image-object sequence to process; None selects collection targets.',
'img_objects': 'Image objects to attach channel data to.',
'load_data': 'Load channel TIFFs before attaching them when true; false uses already loaded field-keyed data.',
'df': 'DataFrame to inspect, transform, or augment. Correlation mutates it in place and returns it.',
'feature_method_tuples': 'Pairs of (feature-name list, correlation-selector list) applied to each channel pair.',
'prepared_feature': 'Optional cached signal arrays and validity mask for a channel/feature pair.',
'channel1': 'First channel prefix for the matching feature.', 'channel2': 'Second channel prefix for the matching feature.',
'feature': 'Unprefixed shared feature name; corresponding channel-prefixed columns must exist.',
'method_name': 'Implemented correlation selector, such as pearson, manders, or ratio.',
'level': 'Logging level name, such as INFO or WARNING.',
'cell': 'Cell instance containing geometry and measurements.',
'image_obj': 'Image instance supplying calibration, geometry, and channel data.',
'image_object': 'Image instance containing the selected field and cells.',
'cell_id': 'Cell identifier local to the selected image; check that it remains after curation/filtering.',
'field_id': 'Shared TIFF field stem before the channel suffix.',
'dataset_dir': 'Folder containing the channel TIFF files.',
'channel_labels': 'Mapping from channel identifiers to human-readable signal labels.',
'crop_size': 'Displayed crop extent in pixels.',
'window_px': 'Cell plot window extent in pixels.',
'show': 'Display generated plots when true; check the documented return expression for figures.',
'contour': 'Boundary points in (row, column) pixel coordinates.',
'midline': 'Central-axis points in (row, column) pixel coordinates.',
'mesh': 'Paired-boundary geometry with four columns in pixel coordinates.',
'labels': '2D integer label mask with zero background.',
'connectivity': 'Touching-label neighborhood: 1/4 for edges, 2/8 including diagonals.',
'max_neighbors': 'Remove labels whose distinct touching-label count is greater than this limit.',
}

OVERRIDES = {
'ImageCollection': ('Create a dataset collection with physical calibration and explicit error handling.', 'Use this as the public entry point; load fields/masks or reload geometry before dependent analysis.'),
'Curation': ('Prepare phase/morphology features and predict retained/rejected cells using a named SVM.', 'Supply the svm feature DataFrame produced by the collection workflow; compiled_curation loads a model, prepares features, and returns labels.'),
'SignalCorrelation': ('Select an implemented relationship between two channel-prefixed feature columns.', 'The feature must exist for both channels. calculate mutates the supplied DataFrame; prepared_feature can cache the numerical input arrays.'),
'Image': ('Represent one microscopy field, its calibrated cell population, and channel caches.', 'Cell geometry and detections are attached during the collection workflow. Array geometry uses row/column pixel coordinates.'),
'Cell': ('Store one cell contour, paired mesh, midline, identity, and feature dictionaries.', 'The cell_id is local to a field; geometry arrays remain in pixel units.'),
'Features': ('Calculate per-cell geometry and signal feature families for an image.', 'Collection and Image methods coordinate the required loading, filtering, feature error recording, and table construction.'),
'load_masks': ('Read label TIFFs from the dataset masks subfolder.', 'Requires a dataset path; modifies masks and mask filenames; returns None.'),
'load_phase_images': ('Load phase TIFFs using the selected channel suffix.', 'Requires a dataset path; sets images, filenames, paths, phase suffix, and stage state; returns None.'),
'load_channel_images': ('Load channels and index their arrays by field identity.', 'Requires a dataset path; clears stale data for requested channels and records load/duplicate-field failures; returns None.'),
'create_image_objects': ('Validate field matching and build a fresh image/cell population.', 'Loads missing phase/mask data, rejects invalid setup or stale masks after segmentation failure, and clears feature tables; returns None.'),
'segment_images': ('Run Omnipose for selected phase images and save their masks.', 'Requires loaded phase images. Returns True on success or False on a recorded failure in continue mode; raise policy propagates errors.'),
'batch_process_mesh': ('Join/split cells, construct geometry, collect mesh tables, and optionally save a pickle.', 'Requires compatible phase images and masks. Updates cells, mesh_df_collection and mesh_prefilter_stats. Returns None.'),
'batch_load_mesh': ('Load saved pickle or mesh Parquet and reconstruct image objects.', 'Requires the matching phase images/masks and compatible geometry columns. Recreates population and clears feature state; returns None.'),
'batch_detect_objects': ('Replace requested-channel detections and collect object contours/geometry.', 'Requires image objects with cells. Loads channels, changes cell.object_meshdata and object_detection_df; returns the combined detection DataFrame.'),
'batch_calculate_features': ('Calculate selected feature families for the current cell population.', 'Requires image objects and loaded signals/detections as appropriate. Replaces selected tables, may remove large cells, clears merged_features; returns the feature_dataframes dictionary.'),
'batch_calculate_signal_correlation_features': ('Add selected between-channel measurements to the supplied dataframe.', 'Requires both prefixed feature columns. Mutates and returns df; does not independently merge or validate an entire workflow.'),
'curate_dataset': ('Predict SVM quality labels and retain the accepted cell population.', 'Requires cells and a compatible named SVM schema. Resets feature tables, updates curated_df and mesh_df_collection, optionally saves geometry; returns None.'),
'export_curated_masks': ('Rasterize retained contours and record overlap with original segmentation labels.', 'Requires image objects with original masks. Writes TIFFs and a provenance CSV; returns (list of mask paths, provenance DataFrame).'),
'merge_dataframes': ('Merge family/channel tables by image_name, cell_id, and frame.', 'Requires feature tables; channel prefixes are added and a first-value pivot is used. Modifies and returns merged_features.'),
'dataframe_to_pkl': ('Save merged features as a pickle in the image folder.', 'Requires merged_features; uses atomic replacement and returns the saved path.'),
'dataframe_to_parquet': ('Save merged features as compressed Parquet in the image folder.', 'Requires merged_features and PyArrow; replaces the target atomically and returns None.'),
'meshdata_to_parquet': ('Save reloadable nested geometry to compressed Parquet.', 'Requires nonempty mesh_df_collection; writes within the image folder and returns None.'),
'processing_summary': ('Report retained stage and cell error history.', 'Returns a new summary dictionary. completed does not imply every stage ran, and cell errors alone do not change that status.'),
'colocalization': ('Incomplete preliminary colocalization implementation.', 'Prepares channel crops but does not complete a measurement; do not use as an analysis feature family.'),
'get_control': ('Randomly select retained and rejected curation examples.', 'Returns two lists of (cell_id, frame) pairs, or (None, None) when either requested class is too small.'),
'calculate': ('Calculate the selected signal relationship row by row.', 'Requires matching prefixed columns. Adds a function-name-suffixed column to df and returns that same DataFrame.'),
'normalize_per_cell': ('Min–max normalize a profile within one cell.', 'Returns an array scaled to [0, 1]; a constant profile returns zeros.'),
'get_avg_distance_from_center': ('Subtract 0.5 from the mean normalized longitudinal object position.', 'Returns a signed mean position offset, not the mean absolute distance from the center.'),
}

def api_pages():
    modules = sorted({a['module'] for a in DATA['api'] if a['module'] != '__init__'})
    api_index = heading('API reference') + ('Use ``ImageCollection`` as the main workflow entry point. The pages below\n'
        'show current signatures extracted from source without importing the analysis\n'
        'dependencies. Each entry links to a local, line-numbered source snapshot.\n'
        'Public workflow operations and internal helpers are separated.\n\n'
        '.. toctree::\n   :maxdepth: 1\n\n')
    api_index += ''.join(f'   {m}\n' for m in modules)
    write('api/index', api_index)
    coverage = []
    for module in modules:
        path = PACKAGE / 'bactoscoop' / (module + '.py')
        src = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(src)
        entries = [a for a in DATA['api'] if a['module']==module]
        body = heading(f'bactoscoop.{module}') + f'.. py:module:: bactoscoop.{module}\n\n'
        if module == 'imagecollection':
            body += 'The public collection coordinates loading, geometry, curation, measurements, and export. See :doc:`../getting-started/quickstart`.\n\n'
        elif module == 'utilities':
            body += 'Low-level numerical and image helpers. Prefer the collection workflow for routine analysis; inspect geometry conventions before calling these directly.\n\n'
        elif module == 'features':
            body += 'Per-cell feature implementations called by Image and ImageCollection. See :doc:`../reference/feature-dictionary`.\n\n'
        for internal in (False, True):
            selected = [a for a in entries if (a['name'].startswith('_') and a['name'] != '__init__') == internal and a['name'] != '__init__']
            if not selected:
                continue
            body += heading('Internal helpers' if internal else 'Classes and public operations', '-')
            for entry in selected:
                cls = entry['class']
                full_name = f'{cls}.{entry["name"]}' if cls else entry['name']
                body += heading(full_name, '~')
                node = next((n for n in ast.walk(tree) if getattr(n, 'lineno', -1)==entry['line'] and isinstance(n,(ast.FunctionDef,ast.ClassDef))), None)
                if entry.get('kind')=='class':
                    ctor = next((m for m in node.body if isinstance(m,ast.FunctionDef) and m.name=='__init__'), None)
                    args = ast.unparse(ctor.args) if ctor else ''
                    args = re.sub(r'^(self|cls)(, )?', '', args)
                    signature = f'{full_name}({args})'
                    directive = 'class'
                    paramnode = ctor
                else:
                    args = re.sub(r'^(self|cls)(, )?', '', entry['args'])
                    signature = f'{full_name}({args})'
                    directive = 'method' if cls else 'function'
                    paramnode = node
                body += f'.. py:{directive}:: {signature}\n\n'
                doc = entry['doc'].strip()
                summary = OVERRIDES.get(entry['name'], (None,None))[0]
                if not summary:
                    summary = re.split(r'\n\s*\n|\nParameters|\nArgs:|\nReturns', doc)[0].strip()
                    summary = re.sub(r'\s+', ' ', summary) if summary else entry['name'].replace('_',' ').capitalize() + '.'
                body += textwrap.indent(field_description(summary) + '\n', '   ') + '\n'
                contract = OVERRIDES.get(entry['name'], (None,None))[1]
                if contract:
                    body += textwrap.indent(contract + '\n', '   ') + '\n'
                if paramnode:
                    params = [p.arg for p in [*paramnode.args.posonlyargs,*paramnode.args.args,*paramnode.args.kwonlyargs] if p.arg not in ['self','cls']]
                    for p in params:
                        description = PARAMETERS.get(p) or documented_parameter(doc,p) or f'Input ``{p}`` consumed by this implementation; its exact use is visible in the linked source below. No maintained parameter description is present in the original docstring.'
                        body += f'   :param {p}: {field_description(description)}\n'
                    if params:
                        body += '\n'
                    returns = list(dict.fromkeys(ast.unparse(n.value) if n.value else 'None' for n in ast.walk(paramnode) if isinstance(n,ast.Return)))
                    if not returns:
                        body += '   :returns: None (no explicit return statement).\n\n'
                    else:
                        body += '   **Return expressions in source:**\n\n'
                        body += textwrap.indent(code('\n'.join(returns)), '   ')
                    mutations = []
                    for n in ast.walk(paramnode):
                        if isinstance(n,ast.Assign):
                            for t in n.targets:
                                for child in ast.walk(t):
                                    if isinstance(child,ast.Attribute) and ast.unparse(child).startswith(('self.','cell.')):
                                        mutations.append(ast.unparse(child))
                    mutations = list(dict.fromkeys(mutations))
                    if mutations:
                        body += '   **Assigned state:** ' + ', '.join(f'``{x}``' for x in mutations) + '.\n\n'
                    exceptions = list(dict.fromkeys(ast.unparse(n.exc.func) for n in ast.walk(paramnode) if isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call)))
                    if exceptions:
                        body += '   **Explicitly raised exceptions:** ' + ', '.join(f'``{x}``' for x in exceptions) + '. Errors from called functions can also propagate or be recorded.\n\n'
                body += f'   `View current source, line {entry["line"]} <../source-code/{module}.html#{module}-{entry["line"]}>`_.\n\n'
                if doc:
                    body += '   .. raw:: html\n\n      <details><summary>Original docstring (legacy wording may differ from the current signature)</summary>\n\n'
                    body += textwrap.indent(code(doc, 'text'), '   ')
                    body += '   .. raw:: html\n\n      </details>\n\n'
                coverage.append({'module':module,'name':full_name,'line':entry['line'],'internal':internal})
        write('api/' + module, body)
        formatted = highlight(src, PythonLexer(), HtmlFormatter(linenos='inline', lineanchors=module, anchorlinenos=True))
        write('source-code/' + module, ':orphan:\n\n' + heading(f'Source: bactoscoop/{module}.py') + '.. raw:: html\n\n' + textwrap.indent('<div class="source-lines">'+formatted+'</div>', '   '))
    (ROOT/'reports/api-coverage.json').write_text(json.dumps(coverage,indent=2), encoding='utf-8')

def spreadsheet_rows():
    categories = {'Morphological':'morphological','Profiling':'profiling','SVM':'svm','Object':'objects','Membrane':'membrane','Signal':'correlation'}
    rows = {}
    category = None
    for row in DATA['spreadsheet']['Sheet1'][1:]:
        if row[0].startswith('###'):
            category = next((v for k,v in categories.items() if k.lower() in row[0].lower()),None)
        elif row[0] and category:
            rows[(category,row[0])] = row
    return rows

def feature_units(category,key, spreadsheet):
    unit = spreadsheet[2] if spreadsheet else 'See implementation'
    unit = unit.replace('�m�','µm²' if 'area' in key else 'µm³').replace('�m','µm')
    if key.endswith('SOV') or key=='cell_SOV':
        return 'µm⁻¹'
    if 'bending_energy' in key or 'bending_energies' in key:
        return 'µm⁻³ under the implemented sum(curvature²)/length'
    if 'sphericity' in key or 'sphericities' in key:
        return 'µm⁻¹ᐟ² under the implemented 2D formula'
    if 'curvature' in key or key in ['cell_avg_obj_mean_c','cell_avg_obj_std_c']:
        return 'µm⁻¹'
    if key in ['l','d','step_length_demograph']:
        return 'µm'
    if key in ['l_norm','d_norm','object_avg_center_distance']:
        return 'Dimensionless'
    if key=='cell_area_asymmetry':
        return 'Dimensionless'
    if 'volume' in key:
        return 'µm³'
    if 'area' in key:
        return 'µm²'
    if any(word in key for word in ['length','perimeter','radius','diameter','distance_from_pole','pole_distances','interobject_distance']) or key in ['avg_pole_obj_distance']:
        return 'µm'
    if ('width' in key and not any(t in key for t in ['variability','constr','rel_constr'])):
        return 'µm'
    if 'normalized' in key or any(t in key for t in ['ratio','eccentric','solidity','convex','circular','compact','sinuos','constr','rel_constr','glcm','kurtosis','skew','homogeneity','extent']):
        return 'Dimensionless (GLCM descriptors use quantized intensity bins)'
    if 'intensity' in key or 'intensities' in key:
        return 'Image intensity units after the family-specific preprocessing'
    return 'Dimensionless' if unit=='/' else unit

def feature_pages():
    rows = spreadsheet_rows()
    categories = ['morphological','profiling','objects','membrane','svm']
    intro = heading('Feature dictionary') + ('Use this dictionary to connect exported columns to their calculation.\n'
        'Names are extracted from current assignments, including tuple assignments;\n'
        'region-property component families are expanded below. Descriptions draw on\n'
        '``SingleCellFeatures.xlsx`` and are qualified where its wording differs from code.\n\n'
        'Morphology uses unprefixed names with channel ``None``. Other families export\n'
        '``<channel>_<key>``, such as ``C3_NC_ratio``. Identity columns are documented\n'
        'in :doc:`data-structures`; correlation columns are in :doc:`../workflow/correlation`.\n\n'
        '.. toctree::\n   :maxdepth: 1\n\n')
    intro += ''.join(f'   features-{c}\n' for c in categories)
    intro += '\nArray-valued columns remain arrays/lists within one cell row. They are not\nseparate rows per object. For missing-value interpretation, see :doc:`../workflow/features`\nand :doc:`../workflow/objects`.\n'
    write('reference/feature-dictionary',intro)
    records = []
    for category in categories:
        unique = {f['key']:f for f in DATA['features'] if f['category']==category}
        page = heading(category.capitalize() + ' features')
        prerequisites = {'morphological':'Cell contour, mesh, midline, phase image, and physical calibration.',
            'profiling':'Loaded matching fluorescence channel and valid cell geometry; channel background subtraction is performed.',
            'objects':'Detected contours for the channel; valid object meshes for mesh-dependent quantities.',
            'membrane':'Loaded channel, valid cell geometry, and detected object contours for that channel; measurements sample the reconstructed cell contour.',
            'svm':'Valid cell geometry and inverse-phase signal; these quantities are used by the curation classifier.'}[category]
        page += '**Prerequisites:** ' + prerequisites + '\n\n'
        page += ('A failed cell calculation omits that family row and records the failure.\nMissing values in a merged table are not automatically zero. Numerical NaN can\nalso arise within a returned measurement. The object retention option explicitly\nsets invalid mesh-dependent aggregates to NaN.\n\n')
        for key,f in unique.items():
            row = rows.get((category,key))
            title = row[3] if row and row[3] else key.replace('_',' ').capitalize()
            desc = row[4] if row else 'This quantity is assigned by the expression shown below; inspect its linked helper implementation for the exact calculation.'
            desc = desc.replace('�','µ')
            if key=='object_avg_center_distance':
                desc = 'Calculation: mean(l_norm) - 0.5. This is a signed offset of the mean longitudinal object position, not an average absolute distance.'
            if key=='avg_pole_obj_distance':
                desc = 'Calculation: mean of the two smallest object-to-pole distances returned by get_pole_object_distances. Inspect obj_pole_distances for all distances.'
            if key=='cell_sphericity' or key=='cell_avg_obj_sphericity' or key=='obj_sphericities':
                desc += '\nThe implementation uses a 2D perimeter/area formula and carries µm^(-1/2); it is not standard dimensionless 3D sphericity.'
            if key.startswith('mean_radius') or key.startswith('median_radius') or key=='max_radius':
                desc = 'Distance from the binary-mask centroid to foreground pixels, summarized as the named statistic. Interior foreground pixels are included; this is not exclusively centroid-to-edge distance.'
            if 'bending_energy' in key or 'bending_energies' in key:
                desc += '\nThe current helper sums sampled curvature squared and divides by physical line length. This is sampling-dependent and is not an arc-length integral.'
            column = key if category=='morphological' else '<channel>_' + key
            shape = 'Scalar'
            if key in ['l','d','l_norm','d_norm','step_length_demograph','axial_intensity','raw_average_mesh_intensity','average_mesh_intensity','normalized_axial_intensity','raw_normalized_average_mesh_intensity','normalized_average_mesh_intensity','radial_intensity_distribution','contour_intensity','normalized_contour_intensity','complemented_contour_intensity'] or f['all_data_only']:
                shape = 'Array/list per cell (per object for detailed object outputs)'
            if key=='contour':
                shape = 'N × 2 geometry array in pixel coordinates'
            page += heading(key,'-') + f'**Exported column:** ``{column}``\n\n**Meaning:** {title}.\n\n'
            page += plain(desc.strip()) + '\n\n'
            page += f'**Units:** {feature_units(category,key,row)}. **Value shape:** {shape}.\n\n'
            page += ('**Availability:** only when ``all_data=True``.\n\n' if f['all_data_only'] else '**Availability:** emitted whenever this family calculation succeeds; ``all_data`` does not gate this key.\n\n')
            page += '**Normalization:** ' + ('Within-cell min–max scaling for normalized axial/mesh/contour profiles; complemented contour intensity is transformed separately in the helper.\n\n' if 'normalized' in key or key=='complemented_contour_intensity' else 'No additional export-level normalization; the calculation below determines family-specific preprocessing.\n\n')
            page += '**Current assignment expression** (a tuple expression can produce several named outputs):\n\n' + code(f['expression'])
            calls = list(dict.fromkeys(re.findall(r'u\.([a-zA-Z_][a-zA-Z0-9_]*)',f['expression'])))
            if calls:
                page += '**Calculation helpers:** ' + ', '.join(f':py:func:`bactoscoop.utilities.{name}`' for name in calls) + '.\n\n'
            page += f'`Source assignment, line {f["line"]} <../source-code/features.html#features-{f["line"]}>`_; :py:meth:`bactoscoop.features.Features.{category}`.\n\n'
            records.append({'category':category,'key':key,'column':column,'line':f['line'],'all_data_only':f['all_data_only']})
        if category=='morphological':
            page += heading('Region-property component families','-')
            page += 'These scalar components are also emitted and are not gated by ``all_data``.\nThey come directly from scikit-image properties in pixel coordinates and do\nnot receive physical calibration. The raw moments and inertia tensor carry\npixel-order units; normalized and Hu moments are dimensionless.\n\n'
            for family in ['moments','moments_central','moments_normalized','moments_hu','inertia_tensor','inertia_tensor_eigvals']:
                keys = [f'{family}_{i}' for i in range(7 if family=='moments_hu' else 2)] if family in ['moments_hu','inertia_tensor_eigvals'] else [f'{family}_{i}_{j}' for i in range(2 if family=='inertia_tensor' else 4) for j in range(2 if family=='inertia_tensor' else 4) if not (family=='moments_normalized' and (i,j) in [(0,0),(0,1),(1,0)])]
                page += heading(family,'~') + 'Columns: ' + ', '.join(f'``{k}``' for k in keys) + '.\n\n'
                page += {'moments':'Raw moments sum row^i × column^j over foreground pixels; units follow the coordinate powers.',
                    'moments_central':'Central moments use coordinates relative to the mask centroid.',
                    'moments_normalized':'Normalized central moments from regionprops; three low-order components are skipped by the utility.',
                    'moments_hu':'Seven Hu moment invariants from the binary region.',
                    'inertia_tensor':'Four 2D inertia tensor components in pixel².',
                    'inertia_tensor_eigvals':'Two inertia tensor eigenvalues in pixel².'}[family] + '\n\n'
                for key in keys:
                    records.append({'category':category,'key':key,'column':key,'dynamic_family':family,'all_data_only':False})
            page += 'Each component is scalar. Degenerate masks or undefined normalized components can produce nonfinite values. See :py:func:`bactoscoop.utilities.get_additional_regionprops_features` for generation and :doc:`../workflow/features` for row-level failure handling.\n'
        write('reference/features-'+category,page)
    (ROOT/'reports/feature-coverage.json').write_text(json.dumps(records,indent=2), encoding='utf-8')

CORRELATIONS = {
'manders': ('manders_overlap_coefficient','Uncentered sum(a*b) / sqrt(sum(a²)*sum(b²)); equal-length profiles; zero energy is undefined.'),
'pearson': ('pearson_correlation_coefficient','Linear correlation via scipy.stats.pearsonr; constant profiles are undefined.'),
'li_icq': ('li_icq','2 × (fraction of positive products of mean-centered signals - 0.5).'),
'ratio': ('ratio','Finite scalar a/b; zero denominator gives NaN.'),
'spearman': ('spearman_rank_correlation','Rank correlation via scipy.stats.spearmanr.'),
'kendall': ('kendall_tau','Rank association via scipy.stats.kendalltau.'),
'distance_corr': ('distance_correlation','Biased sample distance correlation using separately centered pairwise distances; 0–1; constant profiles give NaN.'),
'covariance': ('covariance','Sample covariance np.cov(a,b)[0,1]; retains product intensity units.'),
'n_cross_corr': ('normalized_cross_correlation','Zero-lag mean-centered dot product divided by n*std(a)*std(b). Supply NumPy arrays for subtraction.'),
'entropy_diff': ('entropy_difference','Absolute difference of scipy.stats.entropy values; meaningful for nonnegative distributions.'),
'kurtosis_ratio': ('kurtosis_ratio','Absolute kurtosis(a)/kurtosis(b); zero denominator gives NaN.'),
'skewness_product': ('skewness_product','Product of the two profile skewness statistics.'),
'zero_crossings_diff': ('zero_crossings_difference','Absolute difference in adjacent sign-change counts; raw profiles are not mean-centered.'),
'fft_peak_ratio': ('fft_peak_ratio','Largest FFT magnitude ratio, excluding the DC component; zero denominator gives NaN.'),
'fft_energy_ratio': ('fft_energy_ratio','Ratio of summed squared FFT magnitudes, including DC; zero denominator gives NaN.'),
'histogram_intersection': ('histogram_intersection','Shared-bin, separately probability-normalized histogram overlap; 0–1; unequal lengths allowed.'),
'cosine_similarity': ('cosine_similarity','dot(a,b)/(norm(a)*norm(b)); zero norm is undefined.'),
}

def other_reference():
    deps = tomllib.loads((PACKAGE/'pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies']
    write('generated/dependencies','\n'.join('* ``'+d+'``' for d in deps))
    corr = '.. list-table:: Correlation selectors and output function suffixes\n   :header-rows: 1\n   :widths: 20 30 50\n\n   * - Selector\n     - Output suffix\n     - Calculation / interpretation\n'
    for selector,(function,desc) in CORRELATIONS.items():
        corr += f'   * - ``{selector}``\n     - ``{function}``\n     - {desc}\n'
    write('generated/correlation-methods',corr)
    params = heading('Parameter reference') + 'Signature defaults are listed separately from settings chosen in the tutorials.\nThe tables below are generated from the current source. Follow each method link\nfor prerequisites, returned values, side effects, and source evidence.\n\n'
    for entry in DATA['api']:
        if entry['module']=='imagecollection' and entry['class']=='ImageCollection' and (entry['name']=='__init__' or not entry['name'].startswith('_')):
            name = 'ImageCollection' if entry['name']=='__init__' else entry['name']
            role = 'class' if entry['name']=='__init__' else 'meth'
            full = 'bactoscoop.imagecollection.ImageCollection' + ('' if entry['name']=='__init__' else '.'+name)
            params += heading(name,'-') + f':py:{role}:`{full}`\n\n' + code(name+'('+re.sub(r'^self(, )?','',entry['args'])+')')
            node = ast.parse('def temporary('+entry['args']+'):\n    pass').body[0]
            positional = [*node.args.posonlyargs,*node.args.args]
            defaults = [None]*(len(positional)-len(node.args.defaults)) + node.args.defaults
            for p,default in zip(positional,defaults):
                if p.arg in ['self','cls']:
                    continue
                params += f'* ``{p.arg}`` — '+('required' if default is None else 'default ``'+ast.unparse(default)+'``')+'. '+field_description(PARAMETERS.get(p.arg,'See the linked implementation.'))+'\n'
            params += '\n'
    write('reference/parameters',params)

def adapt_code(code_text, dataset):
    # Notebook sources already use portable paths and isolated working copies.
    return code_text


def tutorial_pages():
    for n in DATA['notebooks']:
        slug = 'three-channel' if '3_channel' in n['file'] else 'five-channel'
        dataset = '3_channel_example' if slug=='three-channel' else '5_channel_example'
        path = PACKAGE / n['file']
        notebook = json.loads(path.read_text(encoding='utf-8'))
        markdown = [f'This walkthrough follows the original `{n["file"].replace(chr(92), "/")}` notebook. '
                    f'Download the [notebook](../_downloads/{slug}.ipynb) or [Python script](../_downloads/{slug}.py). '
                    'The figures show saved results from this example dataset. Run the notebook to reproduce the analysis.\n']
        script = []
        figure_index = 0
        section_caption = 'Saved notebook output'
        for index,cell in enumerate(notebook['cells']):
            content = ''.join(cell['source'])
            if cell['cell_type']=='markdown':
                headings = re.findall(r'^#{1,6}\s+(.+)$', content, flags=re.M)
                if headings:
                    section_caption = re.sub(r'[`*]', '', headings[-1]).strip()
                # Preserve the author's notebook prose rather than imposing first person.
                content = content.replace('[README](../README.md)','[installation guide](../getting-started/installation.rst)')
                content = content.replace('(3_channel_example_walkthrough.ipynb)','(three-channel.md)')
                markdown.append(content.strip()+'\n')
            elif cell['cell_type']=='code':
                content = adapt_code(content,dataset)
                cell['source'] = content.splitlines(keepends=True)
                markdown.append('```python\n'+content.rstrip()+'\n```\n')
                clean = '\n'.join(line for line in content.splitlines() if not line.lstrip().startswith(('%','!')))
                ast.parse(clean)
                script.append(clean)
                for output in cell.get('outputs',[]):
                    if 'image/png' in output.get('data',{}):
                        figure_index += 1
                        name = f'{slug}-{figure_index:02}.png'
                        imagepath = SOURCE/'_static/figures'/name
                        imagepath.parent.mkdir(parents=True,exist_ok=True)
                        png = output['data']['image/png']
                        imagepath.write_bytes(base64.b64decode(''.join(png) if isinstance(png,list) else png))
                        cap = str(output.get('metadata',{}).get('caption') or cell.get('metadata',{}).get('docs_caption') or section_caption).replace('\n',' ')
                        markdown.append(f'```{{figure}} ../_static/figures/{name}\n:alt: {cap}\n\n**{cap}**. Saved notebook output from `{dataset}`; display settings are illustrative.\n```\n')
                cell['outputs'] = []
                cell['execution_count'] = None
        (SOURCE/'examples').mkdir(exist_ok=True)
        (SOURCE/f'examples/{slug}.md').write_text('\n'.join(markdown),encoding='utf-8')
        (SOURCE/f'_downloads/{slug}.py').write_text('\n\n'.join(script)+'\n',encoding='utf-8')
        (SOURCE/f'_downloads/{slug}.ipynb').write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')

def generate():
    (SOURCE/'_downloads').mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'templates/quickstart.py',SOURCE/'_downloads/quickstart.py')
    shutil.copy2(PACKAGE/'BactoScoop Logo.png',SOURCE/'_static/logo.png')
    for name in ['CHANGELOG.md','LICENSE']:
        shutil.copy2(PACKAGE/name,SOURCE/'_downloads'/name)
    # Publish source evidence with relative paths, keeping local administration reports private.
    evidence = dict(DATA)
    evidence['package'] = 'bactoscoop'
    (SOURCE/'_downloads/inspection.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    for name in ['runtime-validation.json','site-validation.json','api-coverage.json','feature-coverage.json']:
        (SOURCE/'_downloads'/name).unlink(missing_ok=True)
    api_pages()
    feature_pages()
    other_reference()
    tutorial_pages()

if __name__=='__main__':
    generate()
