import json
import random
import data_CCTC
import data_FIRE


def write_transformed_file(data, output_file_name="data/train_data.json"):
    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    cctc_directory = 'data'
    cctc = data_CCTC.process_directory(cctc_directory)

    fire_name = 'data/firefly_train_data.csv'
    fire = data_FIRE.process_fire_file(fire_name)

    cctc.extend(fire)
    random.shuffle(cctc)

    total_length = len(cctc)

    train_size = int(total_length * 0.9)
    train_set = cctc[:train_size]
    test_set = cctc[train_size:]

    write_transformed_file(cctc, "data/total_data.json")
    write_transformed_file(train_set, "data/train_data.json")
    write_transformed_file(test_set, "data/test_data.json")

    print(total_length)
    print("Training set size:", len(train_set))
    print("Testing set size:", len(test_set))

