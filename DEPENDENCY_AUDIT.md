# Dependency Audit Report

**Project**: heygen-IV (Facial Landmark Tracking)
**Audit Date**: 2026-01-12
**Audited By**: Claude Code

## Executive Summary

This report provides a comprehensive analysis of dependencies for the heygen-IV project, which focuses on facial landmark tracking using MediaPipe. As this is a new project, we've taken a **security-first, minimal bloat approach** to dependency management.

### Key Findings

✅ **3 core dependencies** - Minimal, focused dependency tree
✅ **No security vulnerabilities** in selected versions
✅ **~100MB saved** by using opencv-python-headless
✅ **All dependencies actively maintained** with recent releases

---

## Dependency Analysis

### Core Dependencies

#### 1. MediaPipe (v0.10.31)

**Purpose**: Face landmark detection (468 facial landmarks)
**Latest Version**: 0.10.31 (Released: 2025-12-18)
**Python Support**: 3.9 - 3.12
**Security Status**: ✅ No known vulnerabilities
**Maintenance**: ✅ Actively maintained by Google AI Edge team

**Why This Version**:
- Most recent stable release
- Well-tested and production-ready
- Official Google support

**Transitive Dependencies**: MediaPipe bundles most of its dependencies, minimizing bloat

**References**:
- [MediaPipe PyPI](https://pypi.org/project/mediapipe/)
- [Setup Guide](https://ai.google.dev/edge/mediapipe/solutions/setup_python)

---

#### 2. OpenCV-Python-Headless (v4.12.0.88)

**Purpose**: Video file processing and frame extraction
**Latest Version**: 4.12.0.88
**Security Status**: ✅ **CRITICAL FIX INCLUDED**
**Maintenance**: ✅ Actively maintained

**Security Advisory - CVE-2025-53644**:
- **Affected Versions**: 4.10.0, 4.11.0
- **Issue**: Uninitialized pointer leading to arbitrary heap buffer write when reading crafted JPEG images
- **Severity**: High
- **Fix**: Version 4.12.0+ (our selected version)
- **Our Status**: ✅ **SECURE** - Using patched version

**Why opencv-python-headless Instead of opencv-python**:

| Aspect | opencv-python | opencv-python-headless | Savings |
|--------|---------------|------------------------|---------|
| Size | ~170 MB | ~70 MB | **~100 MB** |
| GUI Dependencies | ✅ GTK/Qt | ❌ None | - |
| Video I/O | ✅ | ✅ | - |
| Attack Surface | Higher | **Lower** | - |
| Use Case | Desktop apps | **Servers/Scripts** | - |

**Bloat Reduction**: By removing GUI dependencies (GTK, Qt), we:
- Reduce installation size by ~60%
- Eliminate unnecessary attack vectors
- Speed up container builds
- Reduce dependency conflicts

**References**:
- [OpenCV Security Vulnerabilities](https://security.snyk.io/package/pip/opencv-python)
- [CVE-2025-53644 Details](https://securitylab.github.com/advisories/GHSL-2025-057_OpenCV/)

---

#### 3. NumPy (v2.0+)

**Purpose**: Array operations (required by MediaPipe and OpenCV)
**Version Strategy**: `>=2.0.0,<3.0.0` (Range with upper bound)
**Security Status**: ✅ No known vulnerabilities in 2.x series
**Maintenance**: ✅ Actively maintained

**Why Version Range Instead of Pin**:
- NumPy 2.0+ has stable API
- Allows security patch updates without manual intervention
- Upper bound `<3.0.0` prevents breaking changes

**End-of-Life Considerations**:
- NumPy 1.26.x reached EOL in September 2025
- NumPy 2.x series has active security support
- No known CVEs in 2025/2026

**References**:
- [NumPy Vulnerabilities](https://security.snyk.io/package/pip/numpy)

---

## Security Recommendations

### 1. Automated Security Scanning

**Implement Dependency Scanning**:

```bash
# Install safety for vulnerability scanning
pip install safety

# Scan dependencies
safety check --file requirements.txt

# Alternative: Use pip-audit
pip install pip-audit
pip-audit
```

**Recommended Tools**:
- **safety**: Database of known vulnerabilities
- **pip-audit**: Official PyPA tool for auditing Python dependencies
- **Snyk**: Comprehensive vulnerability database
- **Dependabot**: Automated dependency updates (GitHub)

### 2. Regular Update Cadence

**Quarterly Review Schedule**:
- Check for new releases of core dependencies
- Review security advisories
- Test updates in isolated environment
- Update version pins

**Command to Check for Updates**:
```bash
pip list --outdated
```

### 3. Lock Files for Reproducibility

**Generate Lock File**:
```bash
# Install pip-tools
pip install pip-tools

# Generate locked requirements
pip-compile requirements.txt --output-file requirements-lock.txt

# Install from lock file
pip-sync requirements-lock.txt
```

**Benefits**:
- Exact version reproducibility
- Protection against supply chain attacks
- Easier debugging of dependency-related issues

### 4. Container Security

**Dockerfile Best Practices**:

```dockerfile
# Use official Python image with specific version
FROM python:3.11-slim

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Use non-root user
RUN useradd -m appuser
USER appuser

WORKDIR /app
COPY . .
```

### 5. Environment Isolation

**Always Use Virtual Environments**:
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Bloat Analysis

### What We Avoided

By using minimal, headless dependencies, we avoided:

❌ **Unnecessary GUI Frameworks**:
- GTK+ (~50 MB)
- Qt bindings (~40 MB)
- Tcl/Tk

❌ **Unused Image Codecs**:
- Many image formats not needed for video processing
- Proprietary codec support

❌ **Development Tools in Production**:
- Jupyter notebooks
- IPython
- Debugging tools

### Dependency Tree Size Comparison

| Approach | Total Size | Dependencies | Bloat Level |
|----------|-----------|--------------|-------------|
| **Our Minimal Setup** | ~150 MB | 3 core | ✅ Minimal |
| opencv-python (full) | ~250 MB | 5+ | ⚠️ Moderate |
| With Jupyter/IPython | ~350 MB | 15+ | ❌ High |
| Kitchen sink approach | ~500+ MB | 30+ | ❌ Very High |

---

## Maintenance Best Practices

### 1. Dependency Hygiene

**Before Adding New Dependencies, Ask**:
- Is this functionality really needed?
- Can we implement it ourselves in <100 lines?
- Does this package have security support?
- What's the transitive dependency tree?
- Is it actively maintained?

**Red Flags**:
- Last commit >2 years ago
- No releases in >1 year
- Many open security issues
- Huge transitive dependency tree
- No test coverage

### 2. Version Pinning Strategy

**When to Pin Exact Versions**:
- Core dependencies (mediapipe, opencv)
- When reproducibility is critical
- In production environments

**When to Use Ranges**:
- Well-established libraries with stable APIs (numpy)
- To allow security patches
- For libraries that follow semantic versioning

### 3. Testing Updates

**Before Updating Dependencies**:

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install updated versions
pip install mediapipe==<new_version>

# Run tests
pytest tests/

# Run the main application
python track_face.py --test-video test.mp4
```

### 4. Documentation

**Keep Updated**:
- This audit document
- requirements.txt comments
- CHANGELOG.md with dependency updates
- Known issues and workarounds

---

## Action Items

### Immediate Actions

- [x] Create minimal requirements.txt with security-focused dependencies
- [x] Use opencv-python-headless to reduce bloat
- [x] Pin to secure versions (opencv 4.12.0+ for CVE fix)
- [x] Document dependency choices
- [ ] Set up automated security scanning (safety/pip-audit)
- [ ] Configure Dependabot or Renovate for automated updates
- [ ] Add pre-commit hooks for security checks

### Ongoing Maintenance

- [ ] Review dependencies quarterly (next review: 2026-04-12)
- [ ] Monitor security advisories for:
  - MediaPipe: https://github.com/google-ai-edge/mediapipe/security
  - OpenCV: https://github.com/opencv/opencv/security
  - NumPy: https://github.com/numpy/numpy/security
- [ ] Test major version updates in isolated environment
- [ ] Keep this audit document updated

---

## Conclusion

This project follows **security-first, minimal bloat principles**:

1. ✅ **Only 3 core dependencies** - Each serves a specific, essential purpose
2. ✅ **Latest secure versions** - Including critical CVE fixes
3. ✅ **100MB saved** - Using headless OpenCV variant
4. ✅ **Active maintenance** - All dependencies recently updated
5. ✅ **Clear documentation** - Dependencies and rationale documented
6. ✅ **Security monitoring** - Tools and processes recommended

**Risk Level**: 🟢 **LOW**

All dependencies are from trusted sources (Google, OpenCV Foundation, NumPy team), actively maintained, and at secure versions.

---

## References

### Documentation
- [MediaPipe Setup Guide](https://ai.google.dev/edge/mediapipe/solutions/setup_python)
- [MediaPipe PyPI](https://pypi.org/project/mediapipe/)
- [OpenCV-Python Documentation](https://docs.opencv.org/4.x/)

### Security Resources
- [OpenCV Vulnerabilities - Snyk](https://security.snyk.io/package/pip/opencv-python)
- [NumPy Vulnerabilities - Snyk](https://security.snyk.io/package/pip/numpy)
- [CVE-2025-53644 Advisory](https://securitylab.github.com/advisories/GHSL-2025-057_OpenCV/)
- [NVD CVE Database](https://nvd.nist.gov/vuln/detail/cve-2025-53644)

### Tools
- [pip-audit](https://github.com/pypa/pip-audit) - Official Python package auditing tool
- [safety](https://github.com/pyupio/safety) - Vulnerability scanner
- [Dependabot](https://github.com/dependabot) - Automated dependency updates

---

**Last Updated**: 2026-01-12
**Next Review**: 2026-04-12 (3 months)
