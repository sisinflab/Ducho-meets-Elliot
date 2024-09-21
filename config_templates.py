import argparse

# config_elliot = """experiment:
#   backend: pytorch
#   data_config:
#     strategy: fixed
#     train_path: ../data/{{0}}/train_indexed.tsv
#     validation_path: ../data/{{0}}/val_indexed.tsv
#     test_path: ../data/{{0}}/test_indexed.tsv
#     side_information:
#       - dataloader: VisualAttribute
#         visual_features: ../data/{{0}}/visual_embeddings_indexed_{batch_size}/{visual_path}
#       - dataloader: TextualAttribute
#         textual_features: ../data/{{0}}/textual_embeddings_indexed_{batch_size}/{textual_path}
#   dataset: {dataset}
#   top_k: 50
#   evaluation:
#     cutoffs: [ 10, 20, 50 ]
#     simple_metrics: [ Recall, Precision, nDCG, HR ]
#   gpu: 0
#   external_models_path: ../external/models/__init__.py
#   models:
    # external.VBPR:
    #   meta:
    #     hyper_opt_alg: grid
    #     verbose: True
    #     save_weights: False
    #     save_recs: False
    #     validation_rate: 10
    #     validation_metric: Recall@20
    #     restore: False
    #   lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
    #   modalities: ('visual', 'textual')
    #   epochs: 200
    #   factors: 64
    #   batch_size: 1024
    #   l_w: [ 1e-5, 1e-2 ]
    #   comb_mod: concat
    #   loaders: ('VisualAttribute', 'TextualAttribute')
    #   seed: 123
    # external.BM3:
    #   meta:
    #     hyper_opt_alg: grid
    #     verbose: True
    #     save_weights: False
    #     save_recs: False
    #     validation_rate: 10
    #     validation_metric: Recall@20
    #     restore: False
    #   lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
    #   multimod_factors: 64
    #   reg_weight: [ 0.1, 0.01 ]
    #   cl_weight: 2.0
    #   dropout: 0.3
    #   n_layers: 2
    #   modalities: ('visual', 'textual')
    #   loaders: ('VisualAttribute', 'TextualAttribute')
    #   epochs: 200
    #   factors: 64
    #   lr_sched: (1.0,50)
    #   batch_size: 1024
    #   seed: 123
    # external.FREEDOM:
    #   meta:
    #     hyper_opt_alg: grid
    #     verbose: True
    #     save_weights: False
    #     save_recs: False
    #     validation_rate: 10
    #     validation_metric: Recall@20
    #     restore: False
    #   lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
    #   factors: 64
    #   epochs: 200
    #   l_w: [ 1e-5, 1e-2 ]
    #   n_layers: 1
    #   n_ui_layers: 2
    #   top_k: 10
    #   factors_multimod: 64
    #   modalities: ('visual', 'textual')
    #   loaders: ('VisualAttribute', 'TextualAttribute')
    #   mw: (0.1,0.9)
    #   drop: 0.8
    #   lr_sched: (1.0,50)
    #   batch_size: 1024
    #   seed: 123

# """

config_elliot = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{{0}}/train_indexed.tsv
    validation_path: ../data/{{0}}/val_indexed.tsv
    test_path: ../data/{{0}}/test_indexed.tsv
    side_information:
      - dataloader: VisualAttribute
        visual_features: ../data/{{0}}/visual_embeddings_indexed_{batch_size}/{visual_path}
      - dataloader: TextualAttribute
        textual_features: ../data/{{0}}/textual_embeddings_indexed_{batch_size}/{textual_path}
  dataset: {dataset}
  top_k: 20
  evaluation:
    cutoffs: [ 20 ]
    simple_metrics: [ Recall, Precision, nDCG, HR ]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.VBPR:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
      modalities: ('visual', 'textual')
      epochs: 200
      factors: 64
      batch_size: 1024
      l_w: [ 1e-5, 1e-2 ]
      comb_mod: concat
      loaders: ('VisualAttribute', 'TextualAttribute')
      seed: 123
    external.BM3:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
      multimod_factors: 64
      reg_weight: [ 0.1, 0.01 ]
      cl_weight: 2.0
      dropout: 0.3
      n_layers: 2
      modalities: ('visual', 'textual')
      loaders: ('VisualAttribute', 'TextualAttribute')
      epochs: 200
      factors: 64
      lr_sched: (1.0,50)
      batch_size: 1024
      seed: 123
    external.FREEDOM:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
      factors: 64
      epochs: 200
      l_w: [ 1e-5, 1e-2 ]
      n_layers: 1
      n_ui_layers: 2
      top_k: 10
      factors_multimod: 64
      modalities: ('visual', 'textual')
      loaders: ('VisualAttribute', 'TextualAttribute')
      mw: (0.1,0.9)
      drop: 0.8
      lr_sched: (1.0,50)
      batch_size: 1024
      seed: 123
    external.NGCFM:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      lr:  [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
      epochs: 200
      n_layers: 3
      factors: 64
      weight_size: 64
      node_dropout: 0.1
      message_dropout: 0.1
      batch_size: 1024
      modalities: ('visual', 'textual')
      loaders: ('VisualAttribute','TextualAttribute')
      normalize: True
      l_w: [ 1e-5, 1e-2 ]
      seed: 123
    external.GRCN:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      lr: [0.0001, 0.001, 0.01, 0.1, 1]
      epochs: 200
      num_layers: 2
      num_routings: 3
      factors: 64
      factors_multimod: 128
      batch_size: 1024
      aggregation: add
      weight_mode: confid
      pruning: True
      has_act: False
      fusion_mode: concat
      modalities: ('visual', 'textual')
      l_w: [1e-5, 1e-2]
      seed: 123
    external.LATTICE:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      epochs: 200
      batch_size: 1024
      factors: 64
      lr: [0.0001, 0.0005, 0.001, 0.005, 0.01]
      l_w: [1e-5, 1e-2]
      n_layers: 1
      n_ui_layers: 2
      top_k: 20
      l_m: 0.7
      factors_multimod: 64
      modalities: ('visual', 'textual')
      seed: 123
"""

config_ducho = """dataset_path: ./local/data/demo_{dataset}
gpu list: 0

visual:
    items:
        input_path: images
        output_path: visual_embeddings_{batch_size}
        model: [
                {{ model_name: ResNet50,  output_layers: avgpool, reshape: [224, 224], preprocessing: zscore, backend: torch, batch_size: {batch_size}}},
                {{ model_name: ./demos/demo_{dataset}/MMFashion.pt,  output_layers: avgpool, reshape: [224, 224], preprocessing: zscore, backend: torch, batch_size: {batch_size}}},
        ]

textual:
    items:
        input_path: meta.tsv
        item_column: asin
        text_column: description
        output_path: textual_embeddings_{batch_size}
        model: [
            {{ model_name: sentence-transformers/all-mpnet-base-v2,  output_layers: 1, clear_text: False, backend: sentence_transformers, batch_size: {batch_size}}},
          ]

visual_textual:
    items:
        input_path: {{visual: images, textual: meta.tsv}}
        item_column: asin
        text_column: description
        output_path: {{visual: visual_embeddings_{batch_size}, textual: textual_embeddings_{batch_size}}}
        model: [
            {{ model_name: openai/clip-vit-base-patch16, backend: transformers, output_layers: 1, batch_size: {batch_size}}},
            {{ model_name: kakaobrain/align-base, backend: transformers, output_layers: 1, batch_size: {batch_size}}},
            {{ model_name: BAAI/AltCLIP, backend: transformers, output_layers: 1, batch_size: {batch_size}}},
        ]
        
"""


split_config = '''
experiment:
  backend: pytorch
  data_config:
    strategy: dataset
    dataset_path: ../data/{{0}}/reviews.tsv
  splitting:
    save_on_disk: True
    save_folder: ../data/{{0}}_splits/
    test_splitting:
      strategy: random_subsampling
      test_ratio: 0.2
    validation_splitting:
      strategy: random_subsampling
      test_ratio: 0.1
  dataset: {dataset}
  top_k: 20
  evaluation:
    cutoffs: [ 10, 20 ]
    simple_metrics: [ Recall, nDCG ]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    MostPop:
      meta:
        verbose: True
        save_recs: True
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['baby', 'office', 'music', 'toys', 'beauty'], help="Dataset name.", required=True)
    parser.add_argument('--batch_size', type=int, help="Batch size.", required=True)


    args = parser.parse_args()
    
    demo_1 = {
        "visual_path": "torch/ResNet50/avgpool",
        "textual_path": "sentence_transformers/sentence-transformers/all-mpnet-base-v2/1"
    }

    elliot_1 = config_elliot.format(
        batch_size=args.batch_size,
        dataset=args.dataset,
        visual_path=demo_1["visual_path"],
        textual_path=demo_1["textual_path"]
        )
    
    elliot_dir = f"./config_files/{args.dataset}_1_{args.batch_size}.yml"
    with open(elliot_dir, 'w') as conf_file:
        conf_file.write(elliot_1)

    del elliot_1, demo_1, elliot_dir
    
    demo_2 = {
        "visual_path": "transformers/openai/clip-vit-base-patch16/1",
        "textual_path": "transformers/openai/clip-vit-base-patch16/1"
    }

    elliot_2 = config_elliot.format(
        batch_size=args.batch_size,
        dataset=args.dataset,
        visual_path=demo_2["visual_path"],
        textual_path=demo_2["textual_path"]
        )
    
    elliot_dir = f"./config_files/{args.dataset}_2_{args.batch_size}.yml"
    with open(elliot_dir, 'w') as conf_file:
        conf_file.write(elliot_2)

    del elliot_2, demo_2, elliot_dir
    
    demo_3 = {
        "visual_path": "torch/MMFashion/avgpool",
        "textual_path": "sentence_transformers/sentence-transformers/all-mpnet-base-v2/1"
    }

    elliot_3 = config_elliot.format(
        batch_size=args.batch_size,
        dataset=args.dataset,
        visual_path=demo_3["visual_path"],
        textual_path=demo_3["textual_path"]
        )
    
    elliot_dir = f"./config_files/{args.dataset}_3_{args.batch_size}.yml"
    with open(elliot_dir, 'w') as conf_file:
        conf_file.write(elliot_3)

    del elliot_3, demo_3, elliot_dir

    demo_4 = {
        "visual_path": "transformers/kakaobrain/align-base/1",
        "textual_path": "transformers/kakaobrain/align-base/1"
    }

    elliot_4 = config_elliot.format(
        batch_size=args.batch_size,
        dataset=args.dataset,
        visual_path=demo_4["visual_path"],
        textual_path=demo_4["textual_path"]
        )
    
    elliot_dir = f"./config_files/{args.dataset}_4_{args.batch_size}.yml"

    with open(elliot_dir, 'w') as conf_file:
        conf_file.write(elliot_4)

    del elliot_4, demo_4, elliot_dir
    
    demo_5 = {
        "visual_path": "transformers/BAAI/AltCLIP/1",
        "textual_path": "transformers/BAAI/AltCLIP/1"
    }

    elliot_5 = config_elliot.format(
        batch_size=args.batch_size,
        dataset=args.dataset,
        visual_path=demo_5["visual_path"],
        textual_path=demo_5["textual_path"]
        )
    
    elliot_dir = f"./config_files/{args.dataset}_5_{args.batch_size}.yml"

    with open(elliot_dir, 'w') as conf_file:
        conf_file.write(elliot_5)

    del elliot_5, demo_5, elliot_dir




    split_dir = f"./config_files/split_{args.dataset}.yml"
    with open(split_dir, 'w') as conf_file:
        conf_file.write(split_config.format(dataset=args.dataset))

    del split_dir
    
    ducho = config_ducho.format(
        dataset=args.dataset,
        batch_size=args.batch_size
    )

    ducho_dir = f"./Ducho/demos/demo_{args.dataset}/config.yml"
    with open(ducho_dir, 'w') as conf_file:
        conf_file.write(ducho)
