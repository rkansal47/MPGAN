import os


datasets = ["g", "q", "t"]
samples_dict = {key: {} for key in datasets}
real_samples = {}
num_samples = 50000

# mapping folder names to gen and disc names in the final table
model_name_map = {
    "fc": ["FC", "FC"],
    "fcmp": ["FC", "MP"],
    "fcpnet": ["FC", "PointNet"],
    "graphcnn": ["GraphCNN", "FC"],
    "graphcnnmp": ["GraphCNN", "MP"],
    "graphcnnpnet": ["GraphCNN", "PointNet"],
    "mp": ["MP", "MP"],
    "mpfc": ["MP", "FC"],
    "mplfc": ["MP-LFC", "MP"],
    "mppnet": ["MP", "PointNet"],
    "treeganfc": ["TreeGAN", "FC"],
    "treeganmp": ["TreeGAN", "MP"],
    "treeganpnet": ["TreeGAN", "PointNet"],
}


# Load samples

# models_dir = "/graphganvol/MPGAN/trained_models/"
models_dir = "./trained_models/"

for dir in os.listdir(models_dir):
    if dir == ".DS_Store" or dir == "README.md":
        continue

    model_name = dir.split("_")[0]

    if model_name != "mp":
        if model_name.startswith("mp"):
            print(dir)
            with open(f"{models_dir}/{dir}/args.txt", "r") as f:
                args = eval(f.read())

            args["model"] = "old_mpgan"

            with open(f"{models_dir}/{dir}/args.txt", "w") as f:
                f.write(str(args))

        elif model_name.endswith("mp") and not model_name.startswith("treegan"):
            print(dir)
            with open(f"{models_dir}/{dir}/args.txt", "r") as f:
                args = eval(f.read())

            args["model_D"] = "old_mpgan"

            with open(f"{models_dir}/{dir}/args.txt", "w") as f:
                f.write(str(args))
