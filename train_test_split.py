import splitfolders 

trainPrecentage = 0.8
outPath = "dataset"
inputPath = "data\\clean"

def main():
    splitfolders.ratio(inputPath, output=outPath, seed=42, ratio=(trainPrecentage,1-trainPrecentage), group_prefix=None)

if __name__ == '__main__':
    main()
