# ML_cuda

Insall with

```
tmp_dir=$(mktemp -d)
git clone https://github.com/AndyBarcia/ML_cuda.git "$tmp_dir"
sudo pip install "$tmp_dir/fused_mlp_attn_kernel"
rm -rf "$tmp_dir"
```

## TODO

- Dropout
- Support query-key bias.
- Fix NaN output when mask is 0 for query.
- Optimize. Currently implementation is naive