const { defineConfig, devices } = require('@playwright/test');

// The suite drives the stable track by default. `reuseExistingServer` is a
// footgun once two tracks exist -- it will happily attach to whatever is
// already on port 4000, including a server rendering the other baseurl -- so
// it is allowed for local iteration only.
const BASE_URL = process.env.VAFT_DOCS_URL || 'http://127.0.0.1:4000/vaft/';

module.exports = defineConfig({
  testDir: './tests/visual',
  timeout: 30_000,
  workers: 1,
  fullyParallel: false,
  forbidOnly: true,
  reporter: 'list',
  snapshotPathTemplate: '{testDir}/__screenshots__/{projectName}/{testFilePath}/{arg}{ext}',
  expect: {
    timeout: 5_000,
    toHaveScreenshot: {
      animations: 'disabled',
      caret: 'hide',
      maxDiffPixelRatio: 0.01,
    },
  },
  use: {
    baseURL: BASE_URL,
    colorScheme: 'light',
    locale: 'en-US',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'bundle exec jekyll serve --host 127.0.0.1 --port 4000 --no-watch',
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: 'desktop-chromium',
      use: {
        browserName: 'chromium',
        viewport: { width: 1440, height: 900 },
        permissions: ['clipboard-read', 'clipboard-write'],
      },
    },
    {
      name: 'desktop-webkit',
      use: {
        browserName: 'webkit',
        viewport: { width: 1440, height: 900 },
      },
    },
    {
      name: 'mobile-chromium',
      use: {
        ...devices['Pixel 5'],
        browserName: 'chromium',
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 1,
        hasTouch: true,
        isMobile: true,
        permissions: ['clipboard-read', 'clipboard-write'],
      },
    },
    {
      name: 'mobile-webkit',
      use: {
        ...devices['iPhone 13'],
        browserName: 'webkit',
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 1,
        hasTouch: true,
        isMobile: true,
      },
    },
  ],
});
