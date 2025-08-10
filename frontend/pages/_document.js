import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" />
        <link href="https://fonts.googleapis.com/css2?family=Courier+New:wght@400;700&display=swap" rel="stylesheet" />
        <title>TrueSightQ - AI Content Detector</title>
        <meta name="description" content="Advanced AI content detection system for text, images, audio, and video" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  )
} 