import { execSync } from 'node:child_process';

const run = (cmd) => {
  console.log(`Running: ${cmd}`);
  execSync(cmd, { stdio: 'inherit' });
};

console.log('🔧 Running comprehensive project fixes...');

console.log('\n📝 Fixing code formatting...');
run('npx prettier --write "src/**/*.{ts,tsx,js,jsx,json,md}"');

console.log('\n🔍 Auto-fixing ESLint issues...');
run('npx eslint src --ext .ts,.tsx --fix');

console.log('\n✅ Running TypeScript compilation check...');
run('npx tsc --noEmit');

console.log('\n🎉 All fixes complete!');
