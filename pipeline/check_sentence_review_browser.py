import json
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent/'outputs_sentence_windows_v2/review_preview_v1'
URL='http://127.0.0.1:8875/sentence_windows_review/'
with sync_playwright() as pw:
 browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1080});errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(URL,wait_until='domcontentloaded');page.wait_for_function('rows.length===105 && video.readyState>=2',timeout=60000)
 assert page.locator('.entry').count()==105
 assert page.evaluate('video.paused')
 page.locator('#group').select_option('clean');assert page.locator('.entry').count()==30
 page.locator('#pair').click();page.wait_for_function("mode==='pair' && video.readyState>=2 && !video.paused")
 assert page.evaluate('video.duration<=10')
 page.locator('#end').click();page.wait_for_function("mode==='context' && video.readyState>=2 && !video.paused")
 assert page.evaluate('Math.abs(base()+video.currentTime-(current.end_s-3))<1')
 page.wait_for_function('video.paused',timeout=10000)
 assert page.evaluate('Math.abs(base()+video.currentTime-(current.end_s+3))<.5')
 page.locator('#start').click();page.wait_for_function('!video.paused')
 assert page.evaluate('Math.abs(base()+video.currentTime-Math.max(current.context_start_s,current.start_s-3))<1')
 page.locator('#audio').select_option('listener');page.locator('#gain').select_option('4')
 assert page.evaluate('ctx.state')=='running'
 assert page.evaluate('gains.map(g=>g.gain.value)')==[0,0,4,4]
 page.locator('#audio').select_option('speaker');assert page.evaluate('gains.map(g=>g.gain.value)')==[4,4,0,0]
 page.locator('#audio').select_option('both');page.locator('#gain').select_option('1');assert page.evaluate('gains.map(g=>g.gain.value)')==[1,0,0,1]
 word=page.locator('#listenerWords button').first;target=float(word.get_attribute('data-a'));word.click();page.wait_for_function('!video.paused')
 assert abs(page.evaluate('base()+video.currentTime')-max(target-.25,page.evaluate('current.context_start_s')))<1
 page.locator('#verdict').select_option('仍需讨论');page.locator('#note').fill('自动化验证：测试备注');page.locator('#policy').fill('自动化验证：分类建议')
 with page.expect_download() as dl:page.locator('#export').click()
 exported=json.loads(Path(dl.value.path()).read_text());assert exported['reviews'][0]['note']=='自动化验证：测试备注';assert exported['category_policies']['clean']=='自动化验证：分类建议'
 page.reload();page.wait_for_function('rows.length===105 && video.readyState>=2')
 assert page.locator('#note').input_value()=='自动化验证：测试备注'
 page.locator('#progress').select_option('done');assert page.locator('.entry').count()==1
 page.locator('#progress').select_option('todo');assert page.locator('.entry').count()==104
 page.locator('#progress').select_option('all');page.evaluate("store={reviews:{},policies:{}};localStorage.removeItem(STORAGE);select(rows[0])")
 page.wait_for_function('video.readyState>=2')
 # All group counts can be checked without requiring not-yet-rendered samples to load.
 for key in page.evaluate('Object.keys(data.categories)'):
  page.evaluate('(key)=>{$("group").value=key;renderList()}',key)
  assert page.locator('.entry').count()==5
  assert page.evaluate('visible.every(r=>r.review_flags.includes($("group").value))')
 page.locator('#group').select_option('clean');page.wait_for_function('video.readyState>=2')
 page.evaluate('video.pause();scrollTo(0,0)');page.screenshot(path=str(OUT/'browser_desktop.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.screenshot(path=str(OUT/'browser_mobile.png'),full_page=True)
 response=page.request.get(URL+page.evaluate('current.pair_preview'),headers={'Range':'bytes=0-127'})
 assert response.status==206 and len(response.body())==128
 assert not errors,errors
 (OUT/'browser_validation.json').write_text(json.dumps(dict(passed=True,entries=105,clean=30,review_groups=15,group_sizes=5,pair_context_playback=True,boundary_seek_and_stop=True,word_seek=True,audio_routing_and_gain=True,notes_persist_and_export=True,progress_filters=True,mobile_no_overflow=True,http_range=True,page_errors=errors),indent=2))
 print('PASS: browser playback, boundary seeks, all 16 groups, audio, persistence/export, mobile, HTTP range')
 browser.close()
