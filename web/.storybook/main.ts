import type { StorybookConfig } from '@storybook/react-vite';
const config: StorybookConfig={core:{disableTelemetry:true},stories:['../src/**/*.stories.tsx'],framework:{name:'@storybook/react-vite',options:{}},
  viteFinal:async config=>({...config,build:{...config.build,outDir:'storybook-static'}})};
export default config;
