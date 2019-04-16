import matplotlib.pyplot as plt


def main():
    run_data_path = r"D:\Users\White\Desktop\Code\colman\NLP\redditMBTI\learning\1555199106.287743-450embd-800hidden-1layer"
    with open(run_data_path, "r") as f:
        epochs_raw = {int(line.split(",", 1)[0]): line.split(",", 1)[1].rstrip() for line in f.readlines()}
        epochs = {}
        for num, data in epochs_raw.items():
            epochs[num] = {}
            epochs[num]["loss"] = float(data.split("|")[0])
            for d in filter(None, data.split("|")[1:]):
                epochs[num][d.split(":")[0]] = tuple(map(float, d.split(":")[1].split(",")[:-1]))
    ordered_epochs = sorted(epochs.items(), key=lambda x: x[0])
    keys = ("loss", "all", "EI", "SN", "TF", "JP")
    for key in keys:
        if key == "loss":
            plt.plot([x[0] for x in ordered_epochs], [x[1]["loss"] for x in ordered_epochs])
        else:
            plt.plot([x[0] for x in ordered_epochs], [x[1][key][0] for x in ordered_epochs])
            plt.plot([x[0] for x in ordered_epochs], [x[1][key][1] for x in ordered_epochs])
            plt.plot([x[0] for x in ordered_epochs], [x[1][key][2] for x in ordered_epochs])

        plt.xlabel('Epochs')
        plt.suptitle(key, fontsize=14)

        #plt.ylabel('Loss')
        plt.show()

if __name__== "__main__":
    main()
