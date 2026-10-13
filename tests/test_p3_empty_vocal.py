import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline21_residual_eval_metrics import classify_empty_vocal


def fixture(tmp_path):
 stem=tmp_path/'sync_work/vocal_tmp/pred25_022_generated_vocals.wav'
 return stem,[dict(status='error',reference='pred25_022_generated',error='ffmpeg failed: '+str(stem))]


def test_missing_stem_with_separator_evidence_is_not_valid_sync(tmp_path):
 stem,values=fixture(tmp_path)
 result=classify_empty_vocal(values,'Warning: stem_source array is near-silent or empty.',tmp_path,'sha',9.3e-5)
 assert result[0]['status']=='no_detectable_vocal' and result[0]['metric_valid'] is False
 assert result[0]['source_rms']==9.3e-5 and 'min_dist' not in result[0]
 assert result[0]['original_error']==values[0]['error'] and values[0]['status']=='error'


def test_unrelated_errors_and_existing_stem_still_fail(tmp_path):
 stem,values=fixture(tmp_path)
 assert classify_empty_vocal(values,'CUDA out of memory',tmp_path,'sha',0) is None
 values2=[dict(values[0],error='different missing file')]
 assert classify_empty_vocal(values2,'stem_source array is near-silent or empty',tmp_path,'sha',0) is None
 stem.parent.mkdir(parents=True);stem.write_bytes(b'present')
 assert classify_empty_vocal(values,'stem_source array is near-silent or empty',tmp_path,'sha',0) is None


def test_success_and_multiple_records_unchanged(tmp_path):
 _,values=fixture(tmp_path)
 assert classify_empty_vocal([dict(status='success')],'',tmp_path,'sha',0) is None
 assert classify_empty_vocal(values*2,'stem_source array is near-silent or empty',tmp_path,'sha',0) is None
