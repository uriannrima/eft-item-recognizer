import { IncomingMessage } from "http";
import https from "https";
import fs from "fs";

import eftItemsJson from "../JSONs/eft-items.json";

const IMAGE_FOLDER = "./train";
const MAX_ITEMS = Number.POSITIVE_INFINITY;
const WAIT_BETWEEN_ITEMS = 1000;
const WAIT_BETWEEN_CHUNKS = 1000;
const ITEM_CHUNK_SIZE = 10;

function chunks<T>(array: T[], chunkSize: number): T[][] {
  return Array.from({ length: Math.ceil(array.length / chunkSize) }, (_, i) =>
    array.slice(i * chunkSize, i * chunkSize + chunkSize)
  );
}

async function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function request(url: string) {
  return new Promise<IncomingMessage>((resolve, reject) => {
    return https
      .get(url, (response) => {
        if (response.statusCode !== 200) {
          return reject(new Error(`${url} returned ${response.statusCode}`));
        } else {
          resolve(response);
        }
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}

async function checkIfFolderExists(folderPath: string) {
  try {
    const folderStatus = await fs.promises.stat(folderPath);
    return true;
  } catch (error) {
    return false;
  }
}

async function main() {
  const {
    data: { items },
  } = eftItemsJson;

  const capped = items.slice(0, MAX_ITEMS);

  const itemChunks = chunks(capped, ITEM_CHUNK_SIZE);

  for (const items of itemChunks) {
    console.log("Waiting for the next chunk.");
    await wait(WAIT_BETWEEN_CHUNKS);

    for (const item of items) {
      const folderExists = await checkIfFolderExists(
        `${IMAGE_FOLDER}/${item.id}`
      );

      if (folderExists) {
        console.log(`Skipping ${item.id} because it already exists.`);
        continue;
      }

      console.log("Waiting between items.");
      await wait(WAIT_BETWEEN_ITEMS);

      const {
        baseImageLink,
        gridImageLink,
        // image8xLink,
        image512pxLink,
        inspectImageLink,
      } = item;

      const imageURLs = [
        baseImageLink,
        gridImageLink,
        // image8xLink,
        image512pxLink,
        inspectImageLink,
      ];

      const requests = imageURLs.map((url) => [url, request(url)] as const);

      const responses = await Promise.allSettled(
        requests.map(async ([url, promise]) => [url, await promise] as const)
      );

      for (const settledResponse of responses) {
        if (settledResponse.status === "rejected") {
          console.error(settledResponse.reason);
          return;
        }

        const {
          value: [url, response],
        } = settledResponse;

        const fileName = url.split("/").pop();
        const folderPath = `${IMAGE_FOLDER}/${item.id}`;

        await fs.promises.mkdir(folderPath, { recursive: true });

        const filePath = `${IMAGE_FOLDER}/${item.id}/${fileName}`;
        const file = fs.createWriteStream(filePath);

        response.pipe(file);

        file.on("finish", () => {
          file.close();
          console.log(`Downloaded ${fileName}`);
        });
      }
    }
  }
}

main().then(console.log).catch(console.error);
