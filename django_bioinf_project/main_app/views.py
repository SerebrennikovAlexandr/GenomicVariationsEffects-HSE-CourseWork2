from biomodel_handler.load_net_integrated_gradients import get_self_input_single_experiment_scores, \
    get_fasta_file_single_experiment_scores
from biomodel_handler.data_loader import update_sequence
from django.http import HttpResponse
import json


# ----- Service functions -----


def create_json_http_request(response_json):
    response = HttpResponse(json.dumps(response_json),
                            content_type="application/json")
    response['Access-Control-Allow-Origin'] = 'http://127.0.0.1:8080'
    response['Access-Control-Allow-Credentials'] = 'true'

    return response


# ----- Main part -----


def index(request):
    return HttpResponse("Привет, мир!")


def self_input_form(request):
    if request.method == "POST":
        try:
            content = request.POST
            response_json = dict()

            # работа с boundary
            boundary = request.content_params['boundary']
            template = "----WebKitFormBoundary"
            if boundary.find(template) != -1:
                boundary = boundary[boundary.find(template) + len(template):]

            # выбираем сборку человеческого генома
            assembly = content['assembly']
            if assembly == "hg19 genome assembly":
                path_to_hg = r"../hg_assemblies/hg19/hg19_"

                # выбираем модель-ткань
                tissue = content['tissue']
                if tissue == "Histone H3K4me3 of CD19 Primary Cells":
                    path_to_net = r"../model_results/H3K4me3/best_model/best_checkpoint.pt"

                    chr = content['chromosome']
                    pos = int(content['position'])
                    nucleo = content['nucleotide']

                    most_features = get_self_input_single_experiment_scores(path_to_hg, path_to_net,
                                                                            chr, pos, nucleo, pic_name_unique=boundary)

                    orig_nuc = update_sequence(chr, path_to_hg)[pos - 1]

                    response_json['tissue'] = tissue
                    response_json['chromosome'] = chr
                    response_json['position'] = pos
                    response_json['orig_nuc'] = orig_nuc
                    response_json['variation'] = nucleo
                    response_json['most_features'] = most_features
                    response_json['pic_name_unique'] = boundary

            response = create_json_http_request(response_json)
            response.status_code = 200

            return response

        except Exception as e:
            print(e)
            response = create_json_http_request({"message": e})
            response.status_code = 500


def file_input_form(request):
    if request.method == "POST":
        try:
            file = request.FILES['file']

            content = request.POST
            response_json = dict()

            # валидация файла
            file_name = file.name
            file_type = file_name[file_name.rfind('.') + 1:].lower()
            if file_type != "fa" and file_type != "fasta":
                raise RuntimeError("Unsupported file format")

            file_readlines = [x.decode('utf-8').strip() for x in file.readlines()]
            if len(file_readlines) > 2:
                raise RuntimeError("Unsupported file format")

            ids = file_readlines[0]
            chr = ids[ids.find('>') + 1:ids.find(':')]
            pos = ids[ids.find(':') + 1:]

            # работа с boundary
            boundary = request.content_params['boundary']
            template = "----WebKitFormBoundary"
            if boundary.find(template) != -1:
                boundary = boundary[boundary.find(template) + len(template):]

            # выбираем сборку человеческого генома
            assembly = content['assembly']
            if assembly == "hg19 genome assembly":

                # выбираем модель-ткань
                tissue = content['tissue']
                if tissue == "Histone H3K4me3 of CD19 Primary Cells":
                    path_to_net = r"../model_results/H3K4me3/best_model/best_checkpoint.pt"

                    most_features = get_fasta_file_single_experiment_scores(path_to_net, file_readlines,
                                                                            pic_name_unique=boundary)

                    response_json['tissue'] = tissue
                    response_json['chromosome'] = chr
                    response_json['position'] = pos
                    response_json['most_features'] = most_features
                    response_json['pic_name_unique'] = boundary

            response = create_json_http_request(response_json)
            response.status_code = 200

            return response

        except Exception as e:
            print(e)
            response = create_json_http_request({"message": e})
            response.status_code = 500
