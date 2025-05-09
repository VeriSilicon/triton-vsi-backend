import torch
import vpex
import triton_vsi_backend

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("vsi", 0)

    @torch.no_grad()
    def test_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(x) + 0.5 * torch.relu(y) - 1.0
        t_view = t[:, :256]
        out = torch.mean(t_view, dim=0, keepdim=False)
        return out

    x = torch.randn(size=(8, 512), dtype=torch.float32)
    y = torch.randn(size=(512,), dtype=torch.float32)
    out_golden = test_func(x, y)

    import torch._inductor.config
    torch._inductor.config.triton.codegen_upcast_to_fp32 = False
    torch._inductor.config.triton.use_block_ptr = True
    torch._inductor.config.triton.prefer_nd_tiling = True
    model_compiled = torch.compile(test_func, fullgraph=True)

    with torch.inference_mode():
        x = x.to(device=device)
        y = y.to(device=device)
        out = model_compiled(x, y)

    out = out.cpu()
    torch.testing.assert_close(out, out_golden)
