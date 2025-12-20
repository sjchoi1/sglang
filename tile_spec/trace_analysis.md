# Vectorized Slicing Logic Analysis

## Test Case: Mixed draft counts [2, 0, 1]

### Inputs
```
draft_counts = [2, 0, 1]
scores = [[0.9, 0.8, 0.7, 0.6],   # Request 0
          [0.5, 0.4, 0.3, 0.2],   # Request 1
          [0.95, 0.85, 0.75, 0.65]] # Request 2
```

### Step-by-step trace

1. **Initial values:**
   ```
   bs = 3
   max_drafts = max([2, 0, 1]) = 2
   ```

2. **TopK per request (k=2):**
   ```
   top_indices = [[0, 1],      # Request 0: indices of top-2 scores
                  [0, 1],      # Request 1: indices of top-2 scores
                  [0, 1]]      # Request 2: indices of top-2 scores
   ```

3. **Vectorized slicing:**
   ```
   total_drafts = 2 + 0 + 1 = 3
   cumsum = [2, 2, 3]
   offsets = [0, 2, 2]
   ```

4. **Repeat interleave:**
   ```
   repeat_interleave(offsets, draft_counts)
   = repeat_interleave([0, 2, 2], [2, 0, 1])
   = [0, 0] + [] + [2]
   = [0, 0, 2]

   repeat_interleave(torch.arange(3), draft_counts)
   = repeat_interleave([0, 1, 2], [2, 0, 1])
   = [0, 0] + [] + [2]
   = [0, 0, 2]
   ```

5. **Local indices:**
   ```
   local_indices = [0, 1, 2] - [0, 0, 2] = [0, 1, 0]
   ```

6. **Gather:**
   ```
   request_indices = [0, 0, 2]
   local_indices = [0, 1, 0]

   all_indices = top_indices[request_indices, local_indices]
               = [top_indices[0, 0], top_indices[0, 1], top_indices[2, 0]]
               = [0, 1, 0]  # These are the original score indices
   ```

7. **Split:**
   ```
   sizes = [2, 0, 1]
   split([0, 1, 0], [2, 0, 1])
   = [[0, 1], [], [0]]
   ```

### Result
- Request 0: gets indices [0, 1] ✓ (wanted 2 drafts)
- Request 1: gets indices [] ✓ (wanted 0 drafts)
- Request 2: gets indices [0] ✓ (wanted 1 draft)

## Conclusion

**The logic is CORRECT!** The key insight:
- `torch.repeat_interleave` with zero counts **skips** those elements entirely
- `torch.split` with zero sizes **creates empty tensors** at those positions
- The two operations balance out perfectly

## Efficiency Analysis

### Current approach (vectorized):
- **1 topk** of size `max_drafts` for all requests: O(bs * n_cand * log(max_drafts))
- **Multiple tensor ops** (cumsum, repeat_interleave, gather): O(total_drafts)
- **Total**: O(bs * n_cand * log(max_drafts) + total_drafts)

### Alternative approach (loop):
```python
for i in range(bs):
    if draft_counts[i] > 0:
        top_k = torch.topk(scores[i], draft_counts[i])
        top_scores_index.append(top_k.indices)
```
- **Per-request topk**: O(sum over i: n_cand * log(draft_counts[i]))
- **No gather overhead**
- **Total**: O(sum over i: n_cand * log(draft_counts[i]))

### Which is more efficient?

**Loop is likely MORE efficient** because:
1. Avoids computing topk for full `max_drafts` when some requests want fewer
2. Example: draft_counts = [1, 1, 1, 64]
   - Vectorized: 4 topk calls with k=64 each
   - Loop: 3 topk with k=1, 1 topk with k=64

**Vectorized advantage:**
- Single kernel launch overhead vs multiple
- Better for uniform draft counts

**When is vectorized better?**
- When draft counts are similar (low variance)
- When bs is small (kernel launch overhead dominates)

For TileSpec with **highly variable per-request draft counts**, the loop approach might be simpler and faster.
