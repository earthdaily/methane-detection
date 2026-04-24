cwlVersion: v1.2
$graph:
  - class: Workflow
    id: main
    label: Sentinel-2 methane enhancement detection
    doc: |
      Searches a STAC API for Sentinel-2 L1C scenes over a bbox and time range,
      pairs matching L2A scenes when available, runs the matched-filter methane
      enhancement detector for each L1C item, and aggregates time-signal outputs.
      The workflow emits a STAC Catalog of generated methane products.

    inputs:
      bbox:
        type: string
        doc: |
          Bounding box as a JSON string "[west, south, east, north]" in EPSG:4326.
          Example: "[-3.67, 40.23, -3.61, 40.29]"
      start_datetime:
        type: string
        default: "2023-01-01T00:00:00Z"
        doc: Search start datetime in ISO 8601 format.
      end_datetime:
        type: string
        default: "2023-01-24T23:59:59Z"
        doc: Search end datetime in ISO 8601 format.
      collection1:
        type: string
        default: sentinel-2-l1c
        doc: Primary STAC collection containing Sentinel-2 L1C scenes.
      collection2:
        type: string
        default: sentinel-2-l2a
        doc: Secondary STAC collection used for L2A pairing.
      cloud_cover:
        type: ["null", double]
        default: 10.0
        doc: Optional maximum eo:cloud_cover value.
      limit:
        type: ["null", int]
        default: 10
        doc: Optional maximum number of L1C scenes to process.
      download_bands_list:
        type: string
        default: '["B11.jp2", "B12.jp2"]'
        doc: JSON list of band asset keys to process.
      skip_viz:
        type: boolean
        default: false
        doc: Skip matplotlib PNG outputs and colormap legends.
      skip_colorized:
        type: boolean
        default: false
        doc: Skip the colorized methane heatmap COG.
      skip_overviews:
        type: boolean
        default: false
        doc: Emit single-resolution GeoTIFF outputs.
      catalog_url:
        type: string
        doc: STAC API endpoint used to search and resolve Sentinel-2 items.

    outputs:
      stac_catalog:
        type: Directory
        doc: |
          STAC Catalog directory. Contains catalog.json, stac_items/<id>.json,
          generated assets under assets/, and aggregate signals under signals/.
        outputSource: run_pipeline/stac_catalog

    steps:
      run_pipeline:
        run: "#run-pipeline"
        in:
          bbox: bbox
          start_datetime: start_datetime
          end_datetime: end_datetime
          collection1: collection1
          collection2: collection2
          cloud_cover: cloud_cover
          limit: limit
          download_bands_list: download_bands_list
          skip_viz: skip_viz
          skip_colorized: skip_colorized
          skip_overviews: skip_overviews
          catalog_url: catalog_url
        out: [stac_catalog]

  - class: CommandLineTool
    id: run-pipeline
    requirements:
      DockerRequirement:
        dockerPull: docker.io/earthdaily/methane-detection:v0.1.0
      InlineJavascriptRequirement: {}
      EnvVarRequirement:
        envDef:
          CATALOG_URL: $(inputs.catalog_url)
      NetworkAccess:
        networkAccess: true
      ResourceRequirement:
        coresMin: 2
        ramMin: 4096

    baseCommand: ["python", "/app/run_pipeline.py"]

    inputs:
      bbox:
        type: string
        inputBinding:
          prefix: --bbox
      start_datetime:
        type: string
        inputBinding:
          prefix: --start_datetime
      end_datetime:
        type: string
        inputBinding:
          prefix: --end_datetime
      collection1:
        type: string
        inputBinding:
          prefix: --collection1
      collection2:
        type: string
        inputBinding:
          prefix: --collection2
      cloud_cover:
        type: ["null", double]
        inputBinding:
          prefix: --cloud_cover
      limit:
        type: ["null", int]
        inputBinding:
          prefix: --limit
      download_bands_list:
        type: string
        inputBinding:
          prefix: --download_bands_list
      skip_viz:
        type: boolean
        inputBinding:
          prefix: --skip-viz
      skip_colorized:
        type: boolean
        inputBinding:
          prefix: --skip-colorized
      skip_overviews:
        type: boolean
        inputBinding:
          prefix: --skip-overviews
      catalog_url:
        type: string

    outputs:
      stac_catalog:
        type: Directory
        outputBinding:
          glob: out
