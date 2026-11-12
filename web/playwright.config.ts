import { defineConfig } from '@playwright/test';
export default defineConfig({testDir:'./e2e',timeout:60000,fullyParallel:false,workers:1,
  use:{baseURL:'http://127.0.0.1:8766',viewport:{width:1600,height:1000},trace:'retain-on-failure',
    channel:process.env.PTM_BROWSER_CHANNEL},
  webServer:{command:'../.venv/bin/python ../scripts/seed_e2e.py && ../.venv/bin/python -m ptm serve ../tmp/e2e-data --port 8766',
    url:'http://127.0.0.1:8766/api/v1/runs',reuseExistingServer:!process.env.CI,timeout:30000}});
