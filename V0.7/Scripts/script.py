import asyncio

import hoydedata_api


def main():
    bounds = [251091.9498,7032567.1219,265107.0872,7042621.2324]
    asyncio.run(
        hoydedata_api.download_elevation_model(
            bounds=bounds,
            resolution_meters=10, 
            output_path="dem.tif"
            )
    )


if __name__ == "__main__":
    main()