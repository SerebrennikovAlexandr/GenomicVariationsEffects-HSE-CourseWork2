from biomodel_handler.load_net_integrated_gradients import get_single_experiment_scores
from biomodel_handler.data_loader import update_sequence
from django.http import HttpResponse
import json


def index(request):
    return HttpResponse("Привет, мир!")

def self_input_form(request):
    if request.method == "POST":
        try:
            content = request.POST
            response_json = dict()

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

                    most_features = get_single_experiment_scores(path_to_hg, path_to_net, chr, pos, nucleo, pic_name_unique=boundary)

                    orig_nuc = update_sequence(chr, path_to_hg)[pos - 1]

                    response_json['tissue'] = tissue
                    response_json['chromosome'] = chr
                    response_json['position'] = pos
                    response_json['orig_nuc'] = orig_nuc
                    response_json['variation'] = nucleo
                    response_json['most_features'] = most_features
                    response_json['pic_name_unique'] = boundary

            res = HttpResponse(json.dumps(response_json),
                content_type="application/json")
            res.status_code = 200
            res['Access-Control-Allow-Origin'] = 'http://127.0.0.1:8080'
            res['Access-Control-Allow-Credentials'] = 'true'

            return res

        except Exception as e:
            res = HttpResponse(json.dumps({"message": e}),
                               content_type="application/json")
            res.status_code = 500
            res['Access-Control-Allow-Origin'] = 'http://127.0.0.1:8080'
            res['Access-Control-Allow-Credentials'] = 'true'
