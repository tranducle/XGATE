import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add COLABCODE to path
sys.path.insert(0, str(Path(r"C:\Users\Tran Duc Le\Documents\RESEARCHAGENTFINAL\projects\XGATE\RELATED_DATA\XGATE_Public\COLABCODE")))
from src.model.SecurityBERT_Model import VanillaSecurityBERT, TinySecurityBERT

def repro():
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"Eval device: {eval_device}")
    
    num_classes = 15
    input_features = 49
    
    # 1. Instantiate
    print("Testing Teacher Model configuration...")
    model = VanillaSecurityBERT(num_classes=num_classes, input_features=input_features)
    
    # 2. Checkpoint path (just use a dummy save/load to simulate weights_only)
    ckpt_path = Path("test_ckpt.pth")
    torch.save(model.state_dict(), ckpt_path)
    
    print("Loading state dict with weights_only=True...")
    state_dict = torch.load(ckpt_path, map_location=eval_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(eval_device)
    model.eval()
    
    # 3. Simulate Latency check (Move to CPU)
    print("Moving to CPU for latency check simulation...")
    model.cpu()
    dummy_cpu = torch.randn(1, input_features, device=cpu_device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_cpu)
    
    # 4. Move back to Eval Device
    print(f"Moving back to {eval_device}...")
    model.to(eval_device)
    model.eval()
    
    # 5. Test Quantization
    print("Testing INT8 Quantization...")
    sys.path.insert(0, str(Path(r"C:\Users\Tran Duc Le\Documents\RESEARCHAGENTFINAL\projects\XGATE\RELATED_DATA\XGATE_Public\COLABCODE")))
    from src.training.xgate_core import quantize_dynamic_int8
    
    try:
        q_model = quantize_dynamic_int8(model)
        print("Model Quantized.")
        # print("Quantized layers:")
        # for name, module in q_model.named_modules():
        #     if "Linear" in str(type(module)):
        #         print(f"  {name}: {type(module)}")
        
        print("Moving quantized model to CPU...")
        q_model.cpu()
        print("Testing quantized forward pass...")
        output_int8 = q_model(dummy_cpu)
        print(f"Success! INT8 Output shape: {output_int8.shape}")
    except Exception as e:
        print(f"Quantization test failed: {e}")
        # Inspect the encoder layer that failed
        try:
            layer = q_model.transformer.layers[0]
            print(f"Layer 0 activation type: {type(layer.activation)}")
            # print(f"Layer 0 linear1 type: {type(layer.linear1)}")
        except:
            pass
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    repro()
