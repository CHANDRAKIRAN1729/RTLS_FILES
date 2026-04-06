Yes, that's absolutely possible and is actually a cleaner approach. Let me look at the current structure to confirm the components.

Searched for files matching `**/*.py`

Read [](file:///home/rck/RTLS/4.RTLS/src/train_vae_obs.py)

Read [](file:///home/rck/RTLS/5.RTLS_CBFS/src/train_vae_cbf.py)

Yes, absolutely! That's actually the **simpler and more modular approach**. Looking at the code:

**Current RTLS structure:**
1. `train_vae.py` - Train VAE on free-space data
2. `train_vae_obs.py` - Train collision classifier with **frozen VAE encoder** (loads pretrained VAE, only trains `fc32`, `fc42`, `fc_obs` layers)

**What you're proposing:**
1. `train_vae.py` - Keep as-is (unchanged)
2. `train_cbf.py` (new) - Train CBF head with **frozen VAE encoder** (same pattern as `train_vae_obs.py`)

This is cleaner than the current `train_vae_cbf.py` which does end-to-end training.

The key modifications needed:

```python
# In train_cbf.py (new file, based on train_vae_obs.py pattern):

# 1. Load pretrained VAE
checkpoint_vae = torch.load(args.pretrained_checkpoint_path)
model.load_state_dict(checkpoint_vae['model_state_dict'], strict=False)

# 2. Only train CBF head parameters (freeze VAE)
cbf_params = list(model.fc32.parameters()) + list(model.fc42.parameters())
for fc in model.fc_cbf:
    cbf_params += list(fc.parameters())
optimizer_cbf = optim.Adam(cbf_params, lr=args.lr_cbf)

# 3. Use CBF loss instead of BCE classifier loss
# L_safe + L_unsafe (+ optionally L_cbf for trajectory constraint)
```

**Advantages of this approach:**
- Simpler training - VAE and CBF trained independently
- Faster CBF experimentation - don't retrain VAE each time
- Same latent space guarantees as original RTLS
- Easier debugging - isolate issues to VAE or CBF

Do you want me to create this simpler `train_cbf.py` script that follows the RTLS classifier training pattern?
