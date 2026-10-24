# Security policy

## Supported versions

Only the latest release gets security fixes. Releases come as one download per `v*` tag; the
current line is v0.2.x. If you are on an older release, update to the latest one first and check
whether the problem is still there.

| Version          | Supported |
| ---------------- | --------- |
| Latest release   | Yes       |
| Anything older   | No        |

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately instead:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the problem, what it affects (the plugin, the setup tool or the trainer), the version
   you tested, and the steps to reproduce it.

You should hear back within a week. If the report is confirmed, a fix goes into the next release
and the advisory is published once that release is out. You will be credited in the advisory
unless you would rather not be.

## Scope

In scope: the GradientDrift plugin, the setup tool, and the trainer and simulator in this
repository, including the release downloads built from it.

Out of scope: AssettoServer itself, Assetto Corsa, and third-party cars and tracks. Report those
to their own maintainers.
