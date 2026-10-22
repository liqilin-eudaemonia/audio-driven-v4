"""Five fixed cases before/after the sentence-prefix repair; shared 8875 server."""
import concurrent.futures,json,shutil
from pathlib import Path
import build_sentence_review_preview as preview
import repair_sentence_starts_v3 as repair
ROOT=Path(__file__).resolve().parent
OUT=repair.OUT/'preview'
ORIGINAL=preview.OUT

def main():
 OUT.mkdir(exist_ok=True);rows=json.loads((repair.OUT/'sample_results.json').read_text())['rows'];data=json.loads((ORIGINAL/'manifest.json').read_text());data.update(revision=repair.REV,rows=rows)
 for r in rows:
  r['context_before_s']=round(r['start_s']-r['context_start_s'],6)
  r['before_preview']=f"before/{r['id']}/pair.mp4"
  (OUT/'before').mkdir(exist_ok=True)
  link=OUT/'before'/r['id']
  if not link.exists():link.symlink_to(ORIGINAL/'media'/r['id'],target_is_directory=True)
  if r['repair_outcome']!='repaired':
   (OUT/'media').mkdir(exist_ok=True);link=OUT/'media'/r['id']
   if not link.exists():link.symlink_to(ORIGINAL/'media'/r['id'],target_is_directory=True)
 preview.OUT=OUT
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  for x in ex.map(preview.render,[r for r in rows if r['repair_outcome']=='repaired']):print(x,flush=True)
 preview.save(OUT/'manifest.json',data)
 shutil.copytree(ORIGINAL/'assets',OUT/'assets',dirs_exist_ok=True)
 html=(ROOT/'sentence_review_preview.html').read_text()
 html=html.replace('Seamless · 话轮切分复查','Seamless · 句首回溯前后对照').replace('30 条自动未标记问题 + 15 类问题各 5 条 · 105 个不同会话','原分类中的 5 条固定样本 · 2 条补齐句首，3 条保留原样待复查')
 html=html.replace('这是抽样审查，不是合格率估计。“未标记”仍需人工判断。问题类别可重叠；原始边界、音量保留，不增加模型或画面筛选。仅当前样本加载视频。','只补同一句的前半句，回应起点和终点保持不变。修复样本允许最长约 12 秒，普通样本仍沿用原上限。音量保持原样。')
 html=html.replace('<div id="labels">','<div id="repairInfo" class="box"></div><div id="labels">')
 html=html.replace('<button id="pair">播放实际片段</button>','<button id="before">播放修复前片段</button><button id="pair">播放当前片段</button>')
 html=html.replace("STORAGE='seamless-sentence-review-v1'","STORAGE='seamless-sentence-start-repair-v3'")
 html=html.replace("source:'outputs_sentence_windows_v2'","source:'outputs_sentence_start_repair_v3/sample_results.json'")
 html=html.replace("function base(){return mode==='pair'?current.start_s:current.context_start_s}","function base(){return mode==='pair'?current.start_s:mode==='before'?current.before.start_s:current.context_start_s}")
 html=html.replace("m==='pair'?current.pair_preview:current.context_preview","m==='pair'?current.pair_preview:m==='before'?current.before_preview:current.context_preview")
 html=html.replace("(mode==='pair'?'实际片段':'前后文')","(mode==='pair'?'当前片段':mode==='before'?'修复前片段':'前后文')")
 html=html.replace("$('pair').onclick=", "$('before').onclick=()=>play('before',current.before.start_s);$('pair').onclick=")
 html=html.replace("function select(r){current=r;", """function select(r){current=r;const reasons={repaired:'已补齐同一句前半句',discontinuous_previous_words:'此前词语与当前片段不连续，不能可靠回溯',substantive_voice_in_recovered_prefix:'回溯会跨入对方正式发言，未自动扩大区间'};$('repairInfo').innerHTML=`<b>${esc(reasons[r.repair_outcome]||r.repair_outcome)}</b><p>修复前 ${f(r.before.start_s)} → ${f(r.end_s)}（${f(r.before.duration_s)}）<br>当前 ${f(r.start_s)} → ${f(r.end_s)}（${f(r.duration_s)}）</p><p class="muted">原文：${esc(r.before.speaker_text)}</p><p>补回：${esc((r.start_recovery?.added_words||[]).map(w=>w[2]).join(' ')||'无')}</p><p class="small">绿色区间对应当前版本；修复前/当前片段按钮可直接对照起音。</p>`;""")
 html=html.replace("'<option value=\"all\">全部 105 条</option><option value=\"clean\">自动未标记问题（30）</option>'+Object.keys(d.categories)","'<option value=\"all\">全部 5 条</option>'+[...new Set(rows.map(r=>r.sample_group))]")
 html=html.replace("复查类别允许重叠；每条只放入一个主展示分组。","保持原 056—060 编号。先对照修复前与当前片段，再听前后文。")
 html=html.replace("$('criteria').textContent=r.sample_group", "$('criteria').textContent=r.sample_group")
 html=html.replace("data.categories[r.sample_group][1];$('sampling')", "'检查补回的首句是否自然完整；未修复样本请结合异常原因判断。';$('sampling')")
 html=html.replace("(r.speaker_trimmed?'保留后部句子':'从话轮起点开始')", "(r.speaker_start_recovered?'已补回同一句前半句':r.speaker_trimmed?'保留后部句子':'原话轮起点，待复查')")
 (OUT/'index.html').write_text(html)
 link=ROOT.parents[2]/'project_code/audio-driven-v5/data_previews/seamless_face_listening_v1/sentence_start_repair'
 if link.exists():assert link.resolve()==OUT
 else:link.symlink_to(OUT,target_is_directory=True)
 print('http://127.0.0.1:8875/sentence_start_repair/',flush=True)
if __name__=='__main__':main()
