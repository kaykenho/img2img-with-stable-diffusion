```mermaid

graph TB
    subgraph "Core Libraries"
        T[transformers]
        D[diffusers]
        TO[torch]
        A[accelerate]
        M[matplotlib]
        H[huggingface_hub]
    end

    subgraph "Stable Diffusion Model"
        U["U-Net Architecture"]
        VAE["Variational Autoencoder (VAE)"]
        CLIP["CLIP Model"]
        LD["Latent Diffusion"]
        
        VAE -->|"Compress to Latent Space"| LD
        LD -->|"Denoise"| U
        CLIP -->|"Text Understanding"| LD
    end

    subgraph "Image Upload & Preprocessing"
        UP["upload_image()"]
        IP["Image Preprocessing"]
        
        UP -->|"Raw Image"| IP
        IP -->|"Resize to 512x512"| PROC
    end

    subgraph "Transformation Pipeline"
        PROC["generate_image_from_input_image()"]
        STR["Strength Parameter"]
        NS["Noise Scheduling"]
        
        PROC -->|"Control Transform"| STR
        STR -->|"Apply Noise"| NS
        NS -->|"Iterative Denoising"| OUT
    end

    subgraph "Output Generation"
        OUT["Image Generation"]
        SAVE["Save Image"]
        DISP["Display Image"]
        
        OUT -->|"generated_from_image.png"| SAVE
        SAVE --> DISP
    end

    subgraph "Technical Optimizations"
        GPU["GPU Utilization"]
        LSE["Latent Space Efficiency"]
        
        GPU -->|"Accelerate Computing"| PROC
        LSE -->|"Optimize Memory"| PROC
    end

    T & D & H -->|"Model Loading"| CLIP
    TO -->|"Tensor Operations"| GPU
    A -->|"Parallelization"| GPU
    M -->|"Visualization"| DISP
