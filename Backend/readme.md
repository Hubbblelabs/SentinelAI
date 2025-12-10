# SentinelAI Backend

This is the backend service for SentinelAI, built with Node.js.

## Features

- RESTful API
- Authentication & Authorization
- Database integration
- Error handling

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18+)
- [npm](https://www.npmjs.com/)

### Installation

```bash
git clone https://github.com/yourusername/SentinelAI.git
cd SentinelAI/Backend
npm install
```

### Running the Server

```bash
npm start
```

### Environment Variables

Create a `.env` file in the root directory:

```
PORT=3000
DB_URI=your_database_uri
JWT_SECRET=your_jwt_secret
```

## Project Structure

```
Backend/
├── src/
│   ├── controllers/
│   ├── models/
│   ├── routes/
│   └── app.js
├── package.json
└── README.md
```

## API Documentation

See [API.md](API.md) for endpoint details.

## License

MIT