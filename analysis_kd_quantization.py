import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl
from sacred import Experiment

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


from vilt.modules import ViLTransformerSS
from meter.modules import METERTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule as MTDataModuleVILT
from meter.datamodules.multitask_datamodule import MTDataModule as MTDataModuleMETER
from quantization_utils import (
    get_quantization_config,
    SmallMTDataModuleVILT,
    SmallMTDataModuleMETER,
    quantize_modules
)

class QuantizationAnalyzer:
    def __init__(self, config: Dict, model_type: str = "meter"):
        self.config = config
        self.model_type = model_type
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        if model_type == "meter":
            self.model = METERTransformerSS(config).to(self.device)
            self.dm = SmallMTDataModuleMETER(config, dist=False, num_samples=100)
        else:
            self.model = ViLTransformerSS(config).to(self.device)
            self.dm = SmallMTDataModuleVILT(config, dist=False, num_samples=100)
            
        self.dm.setup("test")
        self.dataloader = self.dm.test_dataloader()
        
        # Create output directory
        self.output_dir = f"analysis_results/{model_type}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_weight_distributions(self, layer_name: str, precision: int = 4) -> Dict:
        """Analyze weight distributions before and after quantization"""
        
        # Create quantized model
        model_quant = deepcopy(self.model)
        model_quant = quantize_modules(model_quant, [layer_name], precision)
        
        # Get weights
        weights_orig = self._get_layer_weights(self.model, layer_name)
        weights_quant = self._get_layer_weights(model_quant, layer_name)
        
        # Calculate statistics
        stats = {
            "original": {
                "mean": weights_orig.mean().item(),
                "std": weights_orig.std().item(),
                "min": weights_orig.min().item(),
                "max": weights_orig.max().item()
            },
            "quantized": {
                "min": weights_quant.min().item(),
                "max": weights_quant.max().item(),
                "dtype": str(weights_quant.dtype)
            }
        }
        
        # Plot distributions
        self._plot_weight_distributions(weights_orig, weights_quant, layer_name)
        
        return stats
    
    def analyze_activation_distributions(self, layer_name: str, precision: int = 4) -> Dict:
        """Analyze activation distributions before and after quantization"""
        # Get quantization config
        quant_config, _ = get_quantization_config(precision)
        
        # Create quantized model
        model_quant = deepcopy(self.model)
        model_quant = quantize_modules(model_quant, [layer_name], precision)
        
        # Get activations
        self.model.eval()
        model_quant.eval()
        
        activations_orig = []
        activations_quant = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get original activations
                orig_acts = self._get_layer_activations(self.model, batch, layer_name)
                # Take mean across sequence dimension to handle variable lengths
                orig_acts = orig_acts.mean(dim=1)  # [batch_size, hidden_dim]
                activations_orig.append(orig_acts)
                
                # Get quantized activations
                quant_acts = self._get_layer_activations(model_quant, batch, layer_name)
                # Take mean across sequence dimension
                quant_acts = quant_acts.mean(dim=1)  # [batch_size, hidden_dim]
                activations_quant.append(quant_acts)
        
        # Concatenate activations
        activations_orig = torch.cat(activations_orig, dim=0)
        activations_quant = torch.cat(activations_quant, dim=0)
        
        # Calculate statistics
        stats = {
            "original": {
                "mean": activations_orig.mean().item(),
                "std": activations_orig.std().item(),
                "min": activations_orig.min().item(),
                "max": activations_orig.max().item()
            },
            "quantized": {
                "mean": activations_quant.mean().item(),
                "std": activations_quant.std().item(),
                "min": activations_quant.min().item(),
                "max": activations_quant.max().item()
            }
        }
        
        # Plot distributions
        self._plot_activation_distributions(activations_orig, activations_quant, layer_name)
        
        return stats
    
    def analyze_knowledge_distillation(self, layer_name: str, scale_factor: float = 1.0, 
                                    precision: int = 4) -> Dict:
        """Analyze knowledge distillation effects"""
        # Create teacher and student models
        teacher = deepcopy(self.model)
        student = deepcopy(self.model)
        
        # Set scale factor for student
        student.scale_factor = torch.nn.Parameter(torch.tensor(scale_factor))
        
        # Quantize student
        student = quantize_modules(student, [layer_name], precision)
        
        # Get outputs
        teacher.eval()
        student.eval()
        
        outputs_teacher = []
        outputs_student = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get teacher outputs
                teacher_out = self._get_layer_outputs(teacher, batch, layer_name)
                # Take mean across sequence dimension to handle variable lengths
                teacher_out = teacher_out.mean(dim=1)  # [batch_size, hidden_dim]
                outputs_teacher.append(teacher_out)
                
                # Get student outputs
                student_out = self._get_layer_outputs(student, batch, layer_name)
                # Take mean across sequence dimension
                student_out = student_out.mean(dim=1)  # [batch_size, hidden_dim]
                outputs_student.append(student_out)
        
        # Concatenate outputs
        outputs_teacher = torch.cat(outputs_teacher, dim=0)
        outputs_student = torch.cat(outputs_student, dim=0)
        
        # Calculate metrics
        metrics = {
            "mse": torch.nn.functional.mse_loss(outputs_teacher, outputs_student).item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                outputs_teacher.flatten(), outputs_student.flatten(), dim=0
            ).item()
        }
        
        # Plot outputs
        self._plot_kd_outputs(outputs_teacher, outputs_student, layer_name)
        
        return metrics
    
    def _get_layer_weights(self, model: torch.nn.Module, layer_name: str) -> torch.Tensor:
        """Helper function to get layer weights"""
        # First try to find the exact layer name
        for name, param in model.named_parameters():
            if layer_name in name and "weight" in name:
                return param.data
                
        # If not found, try to find the quantized version
        for name, module in model.named_modules():
            if layer_name in name:
                # Check if this is a quantized module with packed params
                if hasattr(module, '_packed_params'):
                    # For quantized modules, we need to call weight() as a function
                    return module.weight().data
                # If not quantized, try to find weight parameter directly
                for param_name, param in module.named_parameters():
                    if "weight" in param_name:
                        return param.data
                        
        # If still not found, print available layers for debugging
        print("Available layers in model:")
        for name, _ in model.named_modules():
            print(f"  {name}")
            
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def _get_layer_activations(self, model: torch.nn.Module, batch: Dict, 
                             layer_name: str) -> torch.Tensor:
        """Helper function to get layer activations"""
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output)
        
        # Register hook
        for name, module in model.named_modules():
            if layer_name in name:
                hook = module.register_forward_hook(hook_fn)
                break
        
        # Forward pass
        model(batch)
        # Remove hook
        hook.remove()
        
        return activations[0]
    
    def _get_layer_outputs(self, model: torch.nn.Module, batch: Dict, 
                          layer_name: str) -> torch.Tensor:
        """Helper function to get layer outputs"""
        outputs = []
        
        def hook_fn(module, input, output):
            outputs.append(output)
        
        # Register hook
        for name, module in model.named_modules():
            if layer_name in name:
                hook = module.register_forward_hook(hook_fn)
                break
        
        # Forward pass
        _ = model(batch)
        
        # Remove hook
        hook.remove()
        
        return outputs[0]
    
    def _plot_weight_distributions(self, weights_orig: torch.Tensor, 
                                 weights_quant: torch.Tensor, layer_name: str):
        """Plot weight distributions"""
        plt.figure(figsize=(10, 6))
        
        # Plot original weights
        sns.histplot(weights_orig.cpu().numpy().flatten(), 
                    label="Original", alpha=0.5)
        
        # Plot quantized weights
        sns.histplot(weights_quant.dequantize().detach().cpu().numpy().flatten(), 
                    label="Quantized", alpha=0.5)
        
        # # Plot quantized weights
        # sns.histplot(weights_quant.int_repr().detach().cpu().numpy().flatten(), 
        #             label="Quantized INT REPR", alpha=0.5)
        
        plt.title(f"Weight Distribution: {layer_name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Count")
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"weights_{layer_name.replace('.', '_')}.png"))
        plt.close()
    
    def _plot_activation_distributions(self, activations_orig: torch.Tensor,
                                     activations_quant: torch.Tensor, layer_name: str):
        """Plot activation distributions"""
        plt.figure(figsize=(10, 6))
        
        # Plot original activations
        sns.histplot(activations_orig.cpu().numpy().flatten(),
                    label="Original", alpha=0.5)
        
        # Plot quantized activations
        sns.histplot(activations_quant.cpu().numpy().flatten(),
                    label="Quantized", alpha=0.5)
        
        plt.title(f"Activation Distribution: {layer_name}")
        plt.xlabel("Activation Value")
        plt.ylabel("Count")
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"activations_{layer_name.replace('.', '_')}.png"))
        plt.close()
    
    def _plot_kd_outputs(self, outputs_teacher: torch.Tensor,
                        outputs_student: torch.Tensor, layer_name: str):
        """Plot knowledge distillation outputs"""
        plt.figure(figsize=(10, 6))
        
        # Create Q-Q plot
        teacher_percentiles = np.percentile(outputs_teacher.cpu().numpy().flatten(), 
                                          np.linspace(0, 100, 100))
        student_percentiles = np.percentile(outputs_student.cpu().numpy().flatten(),
                                          np.linspace(0, 100, 100))
        
        plt.plot(teacher_percentiles, student_percentiles, 'b-', label='Q-Q Plot')
        plt.plot([teacher_percentiles[0], teacher_percentiles[-1]],
                [teacher_percentiles[0], teacher_percentiles[-1]], 'r--', label='Perfect Match')
        
        plt.title(f"Q-Q Plot: {layer_name}")
        plt.xlabel("Teacher Output Percentiles")
        plt.ylabel("Student Output Percentiles")
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"kd_qq_{layer_name.replace('.', '_')}.png"))
        plt.close()

def main():
    # Initialize configurations
    from configs import meter_config_nlvr2, vilt_config_nlvr2
    
    # Create output directory
    os.makedirs("analysis_results", exist_ok=True)
    
    # Analyze METER model
    print("Analyzing METER model...")
    meter_analyzer = QuantizationAnalyzer(meter_config_nlvr2, "meter")
    
    # Analyze sensitive layers
    sensitive_layers = [
        "text_transformer.encoder.layer.2.intermediate.dense",
        "text_transformer.encoder.layer.2.output.dense"
    ]
    
    meter_results = {}
    for layer in sensitive_layers:
        print(f"\nAnalyzing layer: {layer}")
        
        # Weight distribution analysis
        weight_stats = meter_analyzer.analyze_weight_distributions(layer)
        print(f"Weight Statistics:\n{weight_stats}")
        
        # Activation distribution analysis
        activation_stats = meter_analyzer.analyze_activation_distributions(layer)
        print(f"Activation Statistics:\n{activation_stats}")
        
        # Knowledge distillation analysis
        kd_metrics = meter_analyzer.analyze_knowledge_distillation(layer)
        print(f"Knowledge Distillation Metrics:\n{kd_metrics}")
        
        meter_results[layer] = {
            "weight_stats": weight_stats,
            "activation_stats": activation_stats,
            "kd_metrics": kd_metrics
        }
    
    # Analyze ViLT model
    print("\nAnalyzing ViLT model...")
    vilt_analyzer = QuantizationAnalyzer(vilt_config_nlvr2, "vilt")
    
    # Analyze sensitive layers
    sensitive_layers = [
        "transformer.blocks.0.mlp.fc1",
        "transformer.blocks.0.mlp.fc2"
    ]
    
    vilt_results = {}
    for layer in sensitive_layers:
        print(f"\nAnalyzing layer: {layer}")
        
        # Weight distribution analysis
        weight_stats = vilt_analyzer.analyze_weight_distributions(layer)
        print(f"Weight Statistics:\n{weight_stats}")
        
        # Activation distribution analysis
        activation_stats = vilt_analyzer.analyze_activation_distributions(layer)
        print(f"Activation Statistics:\n{activation_stats}")
        
        # Knowledge distillation analysis
        kd_metrics = vilt_analyzer.analyze_knowledge_distillation(layer)
        print(f"Knowledge Distillation Metrics:\n{kd_metrics}")
        
        vilt_results[layer] = {
            "weight_stats": weight_stats,
            "activation_stats": activation_stats,
            "kd_metrics": kd_metrics
        }
    
    # Save results
    import json
    with open("analysis_results/meter_results.json", "w") as f:
        json.dump(meter_results, f, indent=4)
    with open("analysis_results/vilt_results.json", "w") as f:
        json.dump(vilt_results, f, indent=4)

if __name__ == "__main__":
    main() 
    main() 