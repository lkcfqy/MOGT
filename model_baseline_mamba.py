import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_mamba_baseline(device="cuda"):
    """
    为了执行学术论文中最为苛刻的同级别对等测试，
    抽出 HuggingFace 上的官方原生 Mamba 130M 以作为绝对硬指标打擂目标。
    """
    model_id = "state-spaces/mamba-130m-hf"
    print(f"🔄 正在向内存装载对标重型兵器: {model_id} ...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 用 bfloat16 加载，契合 L4 竞技场规则
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        ).to(device)
        
        print("✅ Mamba基线装载完成！")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 装载 Mamba 落败，可能缺少 'transformers>=4.39.0' 或 'mamba-ssm' 包。 错误详情: {e}")
        return None, None

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mamba_model, mamba_tok = get_mamba_baseline(device)
    
    if mamba_model:
        # 抛入一组测试弹药观察反应
        input_text = "The MOGT architecture theoretically solves the"
        inputs = mamba_tok(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = mamba_model.generate(**inputs, max_new_tokens=10)
            
        print("\\n[Mamba 130M 世代反击]:")
        print(mamba_tok.decode(outputs[0], skip_special_tokens=True))
