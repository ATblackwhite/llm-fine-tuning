import json
import data_CCTC
import data_FIRE


def write_transformed_file(data, output_file_name="train_data.json"):
    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    cctc_directory = 'data'
    cctc = data_CCTC.process_directory(cctc_directory)

    fire_name = 'data/firefly_train_data.csv'
    fire = data_FIRE.process_fire_file(fire_name)

    cctc.extend(fire)
    print(len(cctc))
    write_transformed_file(cctc)
