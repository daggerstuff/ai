# Security Vulnerability Remediation Plan
**Repository**: pixelatedempathy/ai
**Date**: January 30, 2026
**Total Vulnerabilities**: 49 (1 critical, 16 high, 25 moderate, 7 low)

## Executive Summary

The main `pyproject.toml` is **clean** with no vulnerabilities. All 49 vulnerabilities are located in subdirectories and lock files, primarily in legacy/experimental model directories.

## Vulnerability Distribution

### By Severity
- **Critical (1)**: Not visible in current scan - requires investigation
- **High (16)**: aiohttp, ecdsa, nbconvert, protobuf, urllib3
- **Moderate (25)**: aiohttp, filelock, fonttools, mlx, transformers, virtualenv, werkzeug
- **Low (7)**: aiohttp

### By Location
| Location | Count | Status |
|----------|-------|--------|
| `models/Kurtis-E1-MLX-Voice-Agent/uv.lock` | ~40 | Legacy model - can be archived |
| `dataset_pipeline/pixelated-training/requirements.txt` | 1 | Update protobuf |
| `training_ready/packages/apex/requirements.txt` | 1 | Update protobuf |
| `api/techdeck_integration/requirements.txt` | 1 | Update ecdsa |
| `pipelines/edge_case_pipeline_standalone/requirements.txt` | 1 | Update nbconvert |

## Remediation Strategy

### Priority 1: Critical & High Severity (Immediate)

#### 1. protobuf (High) - JSON Recursion Depth Bypass
**Affected Files**:
- `dataset_pipeline/pixelated-training/requirements.txt`
- `training_ready/packages/apex/requirements.txt`
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to protobuf >= 4.25.0

#### 2. urllib3 (High) - Decompression Bomb & Unbounded Links
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to urllib3 >= 2.0.0

#### 3. aiohttp (High) - Zip Bomb Vulnerability
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to aiohttp >= 3.9.0

#### 4. ecdsa (High) - Minerva Timing Attack
**Affected Files**:
- `api/techdeck_integration/requirements.txt`

**Action**: Update to ecdsa >= 0.18.0

#### 5. nbconvert (High) - Uncontrolled Search Path
**Affected Files**:
- `pipelines/edge_case_pipeline_standalone/requirements.txt`

**Action**: Update to nbconvert >= 7.0.0

### Priority 2: Moderate Severity (Within 1 week)

#### 6. transformers (Moderate) - ReDoS Vulnerabilities
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to transformers >= 4.40.0

#### 7. filelock (Moderate) - TOCTOU Race Condition
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to filelock >= 3.13.0

#### 8. werkzeug (Moderate) - Windows Special Device Names
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to werkzeug >= 3.0.0

#### 9. mlx (Moderate) - Buffer Overflow & Wild Pointer
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to mlx >= 0.16.0

#### 10. virtualenv (Moderate) - TOCTOU Vulnerabilities
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to virtualenv >= 20.25.0

#### 11. fonttools (Moderate) - Arbitrary File Write
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Update to fonttools >= 4.50.0

### Priority 3: Low Severity (Within 2 weeks)

#### 12. aiohttp (Low) - Multiple Low-Severity Issues
**Affected Files**:
- `models/Kurtis-E1-MLX-Voice-Agent/uv.lock`

**Action**: Addressed by Priority 1 aiohttp update

## Recommended Actions

### Option A: Update All Dependencies (Recommended for Active Projects)
1. Update all vulnerable packages to their secure versions
2. Regenerate lock files
3. Run full test suite to ensure compatibility
4. Deploy to staging for validation

### Option B: Archive Legacy Code (Recommended for Experimental Models)
1. Archive `models/Kurtis-E1-MLX-Voice-Agent/` directory (contains ~40 vulnerabilities)
2. Update remaining active dependencies
3. Document archived models for future reference

### Option C: Hybrid Approach (Balanced)
1. Archive clearly experimental/legacy models
2. Update dependencies in active pipelines and APIs
3. Create security policy for future dependency management

## Implementation Steps

### Phase 1: Immediate (Today)
1. ✅ Analyze vulnerability distribution
2. ⬜ Create dependency update plan
3. ⬜ Update critical/high severity dependencies in active code
4. ⬜ Run security scan to verify fixes

### Phase 2: Short-term (This Week)
1. ⬜ Update moderate severity dependencies
2. ⬜ Archive legacy experimental models
3. ⬜ Update documentation
4. ⬜ Create PR for review

### Phase 3: Long-term (Next Sprint)
1. ⬜ Implement automated dependency scanning in CI/CD
2. ⬜ Establish dependency update policy
3. ⬜ Set up Dependabot for automatic updates
4. ⬜ Create security baseline documentation

## Dependency Update Commands

### For requirements.txt files:
```bash
# Update specific package
uv pip install --upgrade protobuf>=4.25.0
uv pip install --upgrade ecdsa>=0.18.0
uv pip install --upgrade nbconvert>=7.0.0
```

### For uv.lock files:
```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package protobuf
uv lock --upgrade-package urllib3
uv lock --upgrade-package aiohttp
```

## Verification

After updates, verify with:
```bash
# Check for remaining vulnerabilities
gh api repos/pixelatedempathy/ai/dependabot/alerts

# Run local security scan
uv run pip-audit --strict

# Run tests
pytest ai/tests/ -v
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking changes from updates | Medium | Medium | Comprehensive testing |
| Legacy model incompatibility | Low | Low | Archive strategy |
| Deployment issues | Low | High | Staging validation |
| Performance regression | Low | Medium | Benchmarking |

## Success Criteria

- [ ] All critical and high severity vulnerabilities resolved
- [ ] All active code paths tested and validated
- [ ] Legacy/experimental code archived or updated
- [ ] Security scan shows 0 vulnerabilities in active code
- [ ] Documentation updated
- [ ] CI/CD pipeline includes security scanning

## Notes

- Main `pyproject.toml` is clean - no action needed
- Most vulnerabilities are in `models/Kurtis-E1-MLX-Voice-Agent/` (experimental)
- Consider archiving experimental models to reduce maintenance burden
- Implement automated dependency updates to prevent future accumulation

## References

- GitHub Dependabot: https://github.com/pixelatedempathy/ai/security/dependabot
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- Python Security Best Practices: https://docs.python.org/3/library/security_warnings.html
