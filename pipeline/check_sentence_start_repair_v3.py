import json
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent/'outputs_sentence_start_repair_v3/preview'
URL='http://127.0.0.1:8875/sentence_start_repair/'
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1400,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(URL);page.wait_for_function('rows.length===5 && video.readyState>=2')
 assert page.locator('.entry').count()==5 and '全部 5 条' in page.locator('#group').inner_text()
 for id in ['056','057','058','059','060']:
  page.locator(f'.entry[data-id="{id}"]').click();page.wait_for_function('(id)=>current.id===id && video.readyState>=2',arg=id)
  page.locator('#before').click();page.wait_for_function("mode==='before' && video.readyState>=2 && !video.paused")
  assert page.evaluate('Math.abs(video.duration-current.before.duration_s)<.085')
  assert page.evaluate('base()===current.before.start_s')
  page.locator('#pair').click();page.wait_for_function("mode==='pair' && video.readyState>=2 && !video.paused")
  assert page.evaluate('Math.abs(video.duration-current.duration_s)<.085')
  page.locator('#start').click();page.wait_for_function("mode==='context' && video.readyState>=2 && !video.paused")
  assert page.evaluate('Math.abs(base()+video.currentTime-Math.max(current.context_start_s,current.start_s-3))<1')
 page.locator('.entry[data-id="056"]').click();page.wait_for_function('video.readyState>=2')
 assert 'How will we find' in page.locator('#speakerText').inner_text()
 assert '8.613s' in page.locator('#repairInfo').inner_text()
 page.locator('#audio').select_option('speaker');page.locator('#gain').select_option('2');assert page.evaluate('gains.map(g=>g.gain.value)')==[2,2,0,0]
 page.locator('#verdict').select_option('可用')
 with page.expect_download() as dl:page.locator('#export').click()
 data=json.loads(Path(dl.value.path()).read_text());assert data['revision']=='sentence-start-repair-v3' and data['reviews'][0]['id']=='056'
 page.evaluate("localStorage.removeItem(STORAGE);store={reviews:{},policies:{}};select(rows[0]);video.pause();scrollTo(0,0)");page.wait_for_function('video.readyState>=2')
 page.screenshot(path=str(OUT/'browser_desktop.png'),full_page=True);page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.screenshot(path=str(OUT/'browser_mobile.png'),full_page=True)
 assert not errors,errors
 (OUT/'browser_validation.json').write_text(json.dumps(dict(passed=True,all_five_before_after_context_played=True,source_timing=True,audio=True,export=True,mobile_no_overflow=True,page_errors=errors),indent=2))
 browser.close();print('PASS five fixed samples: before/after/context, timing, audio, export, mobile')
