# ASTRO APIROUTE TYPING SOLUTION - RECURRING ISSUE #4

## Problem
Astro 5.x APIRoute type import fails with module resolution errors

## Solution
Create custom type definitions based on actual Astro source code:

```typescript
// Define proper types based on Astro 5 API structure
type AstroAPIContext = {
  request: Request
  cookies: unknown
  url: URL
  params: Record<string, string | undefined>
  site?: URL
  generator: string
}

type APIRoute = (context: AstroAPIContext) => Response | Promise<Response>

// Usage:
export const GET: APIRoute = async (context: AstroAPIContext) => {
  const { request } = context
  // ... implementation
}
```

## Key Rules
- DO NOT use @ts-expect-error or ignore comments
- DO NOT remove the APIRoute type
- Always implement proper typing based on Astro source code
- This solution is based on: node_modules/astro/dist/types/public/common.d.ts

## Occurrence Tracking
This is the 4th time we've encountered this exact issue.
Framework: Astro 5.10.1
Last occurrence: Sun Jul  6 10:51:05 AM EDT 2025

