import eslint from '@eslint/js';
import tseslint from '@typescript-eslint/eslint-plugin';
import typescript from '@typescript-eslint/parser';
import prettier from 'eslint-config-prettier';
import vitest from 'eslint-plugin-vitest';

export default [
  // Base configuration for JavaScript and TypeScript files
  {
    files: ['src/**/*.js', 'src/**/*.cjs', 'src/**/*.mjs', 'src/**/*.ts'],
    languageOptions: {
      parser: typescript,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        project: './tsconfig.json', // Ensure the tsconfig.json includes @types/node
      },
      globals: {
        // Node.js globals
        NodeJS: 'readonly',
        console: 'readonly',
        process: 'readonly',
        Buffer: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        module: 'readonly',
        require: 'readonly',
      },
    },
    plugins: {
      '@typescript-eslint': tseslint,
    },
    rules: {
      ...eslint.configs.recommended.rules,
      ...tseslint.configs.recommended.rules,

      // General JavaScript/TypeScript rules
      'prefer-const': 'warn',
      'no-constant-binary-expression': 'error',
      'no-undef': 'off', // TypeScript handles this better

      // TypeScript-specific rules
      '@typescript-eslint/no-explicit-any': 'off', // Allow use of `any` where needed
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          ignoreRestSiblings: true,
        },
      ],
      '@typescript-eslint/no-unsafe-function-type': 'off',
    },
  },
  // Vitest-specific configuration for test files
  {
    files: [
      'test/**/*.test.js',
      'test/**/*.test.ts',
      'test/**/*.spec.js',
      'test/**/*.spec.ts',
    ],
    plugins: {
      vitest, // Add Vitest plugin
    },
    rules: {
      ...vitest.configs.recommended.rules,
    },
  },
  // Prettier overrides: Ensure it's the last configuration to handle formatting
  prettier,
];
