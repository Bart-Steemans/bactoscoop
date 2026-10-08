"""Three-channel analysis from downloaded examples, with supplied masks."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import bactoscoop

examples = os.environ.get('BACTOSCOOP_EXAMPLES_DIR')
candidates = [Path(examples)] if examples else [candidate for base in (Path.cwd(), *Path.cwd().parents) for candidate in (base/'examples',base)]
source = next((base/'3_channel_example' for base in candidates if (base/'3_channel_example').is_dir()),None)
if source is None:
    raise FileNotFoundError('Download the release examples and run from that folder, or set BACTOSCOOP_EXAMPLES_DIR.')
destination = Path.home()/'bactoscoop_runs'/('3_channel_example_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
shutil.copytree(source,destination)
ic = bactoscoop.ImageCollection(str(destination),px=0.065)
ic.create_image_objects(phase_channel='C1')
ic.batch_process_mesh(phase_channel='C1',join_thresh=4,split_thresh=0.5,CD_width=False,smoothing=0.1,save_data=True)
ic.curate_dataset(str(destination/'DnaN_Timecourse.pkl'),control=False)
ic.batch_detect_objects(channels=['C3'],reset_channels=True,align=False,smoothing=0.1,log_sigma=3,kernel_width=3,min_overlap_ratio=0.001,max_external_ratio=0.3)
ic.load_channel_images(['C2'])
ic.add_channels(ic.image_objects,['C2'],load_data=False)
ic.batch_calculate_features([([None],'morphological'),(['C2','C3'],'profiling'),(['C3'],'objects')],all_data=False,reset=True,max_mesh_size=1000)
table = ic.merge_dataframes(discard_morphological_nan=True)
ic.dataframe_to_parquet()
ic.dataframe_to_pkl()
(destination/'processing_summary.json').write_text(json.dumps(ic.processing_summary(),indent=2),encoding='utf-8')
print(f'{len(table)} retained rows; outputs in {destination}')
