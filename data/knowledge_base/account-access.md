# Account Access And MFA

Source ID: account-access

For multi-factor authentication lockouts, support should verify the customer
email, confirm the account owner, and check whether backup codes were already
issued. The assistant may draft a reset checklist only after customer context
matches the ticket requester.

If the account owner cannot be verified, the case must be escalated for human
identity review. The assistant must not disable MFA directly.
