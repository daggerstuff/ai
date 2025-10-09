---
title: 'HIPAA Backup Encryption Verification Checklist'
description: 'Checklist for verifying AES-256 backup encryption compliance with HIPAA requirements.'
updated: '2024-07-01'
status: 'active'
---

# 🛡️ HIPAA Encryption Verification Checklist for Backups

## 1. Risk Analysis & Documentation

- [ ] Conduct and document a thorough risk analysis identifying threats to ePHI in backups.
- [ ] Justify the use of AES-256 as a safeguard for ePHI in your risk management documentation.
- [ ] If encryption is not used in any context, document the rationale and describe equivalent alternative safeguards.

## 2. Encryption in Transit and at Rest

- [ ] Encrypt all backup data at rest using AES-256.
- [ ] Encrypt all backup data in transit using strong protocols (e.g., TLS 1.2+).
- [ ] Ensure no unencrypted ePHI is stored or transmitted.

## 3. Integrity Controls

- [ ] Implement mechanisms (e.g., cryptographic hashes, checksums) to detect unauthorized alteration or destruction of backup data.
- [ ] Regularly verify backup integrity and log results.

## 4. Key Management

- [ ] Store encryption keys securely, separate from encrypted data.
- [ ] Restrict key access to authorized personnel only.
- [ ] Rotate keys periodically and upon personnel changes or suspected compromise.
- [ ] Document key management policies and procedures.

## 5. Policies & Procedures

- [ ] Maintain written encryption and backup policies as part of your HIPAA documentation.
- [ ] Train staff on encryption procedures and incident response.

---

**Summary:**
HIPAA does not mandate AES-256 specifically, but it is a best practice. The law requires "reasonable and appropriate" safeguards, documented risk analysis, and strong controls for both data at rest and in transit. Your implementation of AES-256 for backups, with secure key management and integrity controls, aligns with HIPAA's addressable encryption specification—provided all steps above are followed and documented.
