Your question is **completely valid and very insightful**. Let me clarify the real difference.

## Look at WHERE λ is placed in each loss:

### VAE Training
```python
Loss = λ × MSE + KL      # λ multiplies the PRIMARY objective (MSE)
```

### Planning
```python
Loss = L_goal + λ_prior × L_prior + λ_collision × L_collision
       ↑ no λ!   ↑ λ here              ↑ λ here
```

---

## The Critical Difference

| | VAE Training | Planning |
|---|---|---|
| **λ multiplies** | PRIMARY objective (MSE) | SECONDARY objectives (prior, collision) |
| **No λ on** | KL | L_goal (the primary objective) |

---

## What happens in VAE Training

```
Loss = λ × MSE + KL
```

- Start: λ=1, Loss = 1 × 0.001 + 5 = 5.001 (KL dominates)
- GECO sees MSE (0.001) > goal (0.0008), so it **increases λ**
- End: λ=4471, Loss = 4471 × 0.0002 + 5 ≈ 0.9 + 5 = 5.9

GECO's job is to **increase λ** until MSE reaches the target. Starting at 1 is fine because GECO will push it up as needed.

---

## What would happen in Planning if λ=1

```
Loss = L_goal + 1 × L_prior + 1 × L_collision
     = 0.5 + 5.0 + 0.5 = 6.0
```

L_prior (5.0) dominates. The optimizer minimizes L_prior first → pushes z toward 0 → robot doesn't reach the goal.

**The problem**: L_goal has no λ to increase! It's stuck at coefficient = 1. So if the secondary terms dominate, the goal can never win.

---

## The Solution in Planning

Make λ_prior and λ_collision **small** so L_goal dominates from the start:

```
Loss = 0.5 + 0.002 × 5.0 + 0.15 × 0.5
     = 0.5 + 0.01 + 0.075 = 0.585
```

Now L_goal (0.5) is the largest term. The optimizer focuses on reaching the target.

---

## Summary

| | VAE | Planning |
|---|---|---|
| **λ controls** | The thing we WANT to optimize (MSE) | The things we want as CONSTRAINTS (prior, collision) |
| **GECO's action** | Increase λ → MSE becomes more important | Increase λ → constraints become more important |
| **Why λ=1 works/doesn't work** | Works because GECO will increase it | Doesn't work because goal has no λ and can't fight back |

Your intuition was right: in VAE training, KL does dominate initially. But GECO fixes this by increasing λ (to 4471). In planning, L_goal cannot be boosted (no λ), so we must keep the other λ values small.
