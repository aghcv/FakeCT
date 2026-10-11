"""Exercise the captured ROI workbench in Chromium with no network or server."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


@unittest.skipUnless(os.environ.get('FAKECT_TEST_BROWSER') == '1',
                     'Set FAKECT_TEST_BROWSER=1 and install the browser test profile')
class RoiBrowserTests(unittest.TestCase):
    def test_offline_control_navigation_content_and_no_script_errors(self):
        # Explicit browser mode requires its dependency; do not turn a broken
        # mandatory browser lane into a misleading optional skip.
        self.assertIsNotNone(importlib.util.find_spec('playwright'))
        from playwright.sync_api import sync_playwright
        from student_demo import initialize
        from train_study import run
        with tempfile.TemporaryDirectory() as temporary:
            project = initialize(Path(temporary)/'demo')
            ini = Path(project['cases'][0]['cohort_ini'])
            run(ini,stage='preview',preview_mode='roi-selection')
            report = ini.parent/'review-r1/report.html'
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={'width':1280,'height':1000})
                errors, network = [], []
                page.on('pageerror',lambda error: errors.append(str(error)))
                page.on('console',lambda message: errors.append(message.text) if message.type=='error' else None)
                page.on('request',lambda request: network.append(request.url) if request.url.startswith(('http:','https:')) else None)
                page.goto(report.as_uri(),wait_until='load')
                self.assertEqual(page.locator('.card:visible').count(),1)
                self.assertTrue(page.locator('#previous').is_disabled())
                self.assertIn('point 1',page.locator('#current-point').inner_text().lower())
                first_metadata=page.locator('#control-metadata-0').inner_text()
                self.assertIn('radius',first_metadata.lower())
                self.assertIn('i',first_metadata.lower())
                page.select_option('#point','2')
                self.assertTrue(page.locator('#control-card-2').is_visible())
                self.assertTrue(page.locator('#control-image-2-attenuation').is_visible())
                self.assertFalse(page.locator('#control-image-2-labels').is_visible())
                page.select_option('#content','labels')
                self.assertTrue(page.locator('#control-image-2-labels').is_visible())
                self.assertFalse(page.locator('#control-image-2-attenuation').is_visible())
                page.select_option('#point','5')
                self.assertTrue(page.locator('#next').is_disabled())
                page.click('#previous')
                self.assertTrue(page.locator('#control-card-4').is_visible())
                page.locator('a[data-jump="0"]').click()
                self.assertTrue(page.locator('#control-card-0').is_visible())
                self.assertEqual(page.locator('.card:visible').count(),1)
                self.assertEqual(page.locator('#control-metadata-0').inner_text(),first_metadata)
                for image in page.locator('img').all():
                    self.assertTrue(image.evaluate('(img) => img.complete && img.naturalWidth > 0'))
                self.assertEqual(network,[])
                self.assertEqual(errors,[])
                browser.close()


if __name__ == '__main__':
    unittest.main()
