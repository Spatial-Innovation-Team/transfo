# Commit messages

All commits must follow the [Conventional Commits v1.0.0-beta.2](https://www.conventionalcommits.org/en/v1.0.0-beta.2/) specification.

## Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

## Types

| Type | When to use |
|------|-------------|
| `feat` | Introduces a new feature (maps to MINOR in semver) |
| `fix` | Patches a bug (maps to PATCH in semver) |
| `docs` | Documentation changes only |
| `style` | Formatting, whitespace — no logic change |
| `refactor` | Code change that is neither a fix nor a feature |
| `perf` | Performance improvement |
| `test` | Adding or correcting tests |
| `chore` | Build process, tooling, dependency updates |

## Breaking changes

Add `BREAKING CHANGE:` at the start of the footer (or body) to signal a breaking API change — this maps to MAJOR in semver. Any type can carry a breaking change.

```
feat(api): rename coordinate transform method

BREAKING CHANGE: `transform_point` is now `apply`
```

## Scope

Use a short noun in parentheses to indicate the affected area of the codebase:

```
fix(graph): handle missing edge transform gracefully
```

## Examples

```
feat: add AffineTransform class with numpy array support
fix(mock): correct inverse matrix for level2 scaling
docs: add usage examples to AGENT.md
chore: replace ipympl with inline matplotlib backend
```
