from retranslate_aug import count_label_num, aug_translate
from fenci_aug import FenCiAug, FenCi

if __name__ == '__main__':
    dir = "../data/"
    ori_file = "train_data.txt"
    aug_file = "aug_" + ori_file
    class_size = count_label_num(dir,"train_data.txt")
    class_proportion = [1]
    for i in range(1,len(class_size)):
        class_proportion.append(round((class_size[0] * 1.0 / (len(class_size)-1)) / class_size[i]))
    print(class_proportion)

    write_file = open(dir + ori_file, 'wb')
    write_file.write("label,ques\n".encode('utf-8'))
    aug_translate(dir, ori_file, write_file, class_proportion)

    ori_dev_file = "validation_data_demo.txt"
    fenci_dev_file = "fenci_" + ori_dev_file

    FenCiAug(dir, ori_file, write_file, class_proportion)
    FenCi(dir, ori_dev_file, fenci_dev_file, True)