import tseslint from 'typescript-eslint';
export default tseslint.config({ignores:['storybook-static/**','node_modules/**','src/api.generated.ts']},
  ...tseslint.configs.recommended,
  {rules:{'@typescript-eslint/no-unused-vars':['error',{argsIgnorePattern:'^_',varsIgnorePattern:'^_'}],
    '@typescript-eslint/no-empty-object-type':'off'}});
